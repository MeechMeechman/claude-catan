"""
PPO training loop with action masking for Catan.

Implements CleanRL-style PPO with:
- GAE advantage estimation
- Action masking (invalid action logits set to -inf)
- Potential-based reward shaping
- Opponent pool for self-play diversity
- Wandb logging
- Multiprocessing vectorized environments for GPU efficiency
- Pinned memory + async GPU transfers
"""

import os
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from catan_ai.models.policy import CatanPolicy


@dataclass
class PPOConfig:
    # Training
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    lr_schedule: str = "linear"  # linear | constant
    num_envs: int = 16  # more envs = better GPU utilization
    num_steps: int = 512  # steps per env per rollout
    gamma: float = 0.997
    gae_lambda: float = 0.93
    num_epochs: int = 4
    num_minibatches: int = 8
    clip_coef: float = 0.12
    clip_vloss: bool = True
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.015
    normalize_advantages: bool = True

    # Environment
    num_players: int = 2
    map_type: str = "BASE"
    vps_to_win: int = 10
    reward_type: str = "shaped"  # sparse | shaped

    # Self-play
    opponent_pool_size: int = 30
    opponent_update_freq: int = 50_000
    opponent_latest_prob: float = 0.5
    opponent_pool_prob: float = 0.3
    opponent_heuristic_prob: float = 0.2

    # Model
    hidden_dim: int = 128
    encoder_layers: int = 4
    encoder_output_dim: int = 256

    # Logging
    wandb_project: str = "catan-ai"
    wandb_entity: Optional[str] = None
    log_freq: int = 1
    save_freq: int = 100_000
    checkpoint_dir: str = "checkpoints"

    # Device & performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    use_multiprocess_env: bool = True  # use SubprocVecEnv for parallel stepping
    compile_model: bool = False  # torch.compile (PyTorch 2.x)


class RolloutBuffer:
    """Stores rollout data for PPO updates.

    GPU optimization: buffer lives on GPU. Observations are transferred
    in bulk from CPU numpy arrays using pinned memory for async transfers.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shapes: Dict, action_space_size: int, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.pos = 0

        self.obs = {}
        for key, shape in obs_shapes.items():
            self.obs[key] = torch.zeros((num_steps, num_envs, *shape), device=device)

        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

    def add(self, obs: Dict, action, log_prob, reward, done, value):
        for key in self.obs:
            self.obs[key][self.pos] = obs[key]
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1

    def reset(self):
        self.pos = 0

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, gae_lambda: float):
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_batches(self, num_minibatches: int):
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // num_minibatches
        indices = np.random.permutation(batch_size)

        flat_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in self.obs.items()}
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            mb_obs = {k: v[mb_idx] for k, v in flat_obs.items()}
            yield (
                mb_obs,
                flat_actions[mb_idx],
                flat_log_probs[mb_idx],
                flat_advantages[mb_idx],
                flat_returns[mb_idx],
                flat_values[mb_idx],
            )


class OpponentPool:
    """Maintains a pool of past policy checkpoints for diverse self-play."""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.checkpoints: List[Dict] = []
        self.elo_ratings: List[float] = []

    def add(self, state_dict: Dict, elo: float = 1000.0):
        self.checkpoints.append({k: v.cpu().clone() for k, v in state_dict.items()})
        self.elo_ratings.append(elo)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)
            self.elo_ratings.pop(0)

    def sample(self) -> Optional[Dict]:
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)

    def __len__(self):
        return len(self.checkpoints)


class PPOTrainer:
    """Main PPO training orchestrator.

    GPU efficiency optimizations:
    1. SubprocVecEnv: game stepping runs in parallel worker processes,
       so CPU-bound catanatron logic doesn't block the GPU.
    2. Bulk tensor transfers: observations are stacked as numpy arrays
       in the main process, then transferred to GPU once per step.
    3. Pinned memory: CPU->GPU transfers use non_blocking=True.
    4. Large batches: num_envs=16+ ensures minibatches are big enough
       to saturate GPU compute.
    5. Optional torch.compile: fuses encoder operations for ~20% speedup.
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.use_cuda = "cuda" in str(self.device)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(config.seed)

        # Create environments (multiprocess or in-process)
        self._init_envs()

        # Create policy
        self.policy = CatanPolicy(
            hex_input_dim=self.hex_feat_dim,
            vertex_input_dim=self.vertex_feat_dim,
            edge_input_dim=self.edge_feat_dim,
            player_input_dim=self.player_feat_dim,
            action_space_size=self.action_space_size,
            num_tiles=self.num_tiles,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            hidden_dim=config.hidden_dim,
            encoder_layers=config.encoder_layers,
            encoder_output_dim=config.encoder_output_dim,
        ).to(self.device)

        # Optional torch.compile for fused kernels
        if config.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.policy = torch.compile(self.policy)

        print(f"Policy parameters: {self.policy.count_parameters():,}")

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

        # Static adjacency tensors on GPU
        self.vertex_to_hex = torch.tensor(self._vertex_to_hex_adj, dtype=torch.long, device=self.device)
        self.hex_to_vertex = torch.tensor(self._hex_to_vertex_adj, dtype=torch.long, device=self.device)
        self.vertex_adj = torch.tensor(self._vertex_adj, dtype=torch.long, device=self.device)

        # Rollout buffer on GPU
        obs_shapes = {
            "hex_features": (self.num_tiles, self.hex_feat_dim),
            "vertex_features": (self.num_nodes, self.vertex_feat_dim),
            "edge_features": (self.num_edges, self.edge_feat_dim),
            "player_features": (self.player_feat_dim,),
            "action_mask": (self.action_space_size,),
        }
        self.buffer = RolloutBuffer(
            config.num_steps, config.num_envs, obs_shapes, self.action_space_size, self.device
        )

        self.opponent_pool = OpponentPool(config.opponent_pool_size)

        # CUDA stream for async obs transfer
        self._transfer_stream = torch.cuda.Stream() if self.use_cuda else None

        # Stats
        self.global_step = 0
        self.win_count = 0
        self.loss_count = 0
        self.game_count = 0

    def _init_envs(self):
        """Initialize vectorized or in-process environments."""
        cfg = self.config
        env_config = {
            "num_players": cfg.num_players,
            "map_type": cfg.map_type,
            "vps_to_win": cfg.vps_to_win,
            "reward_type": cfg.reward_type,
        }

        if cfg.use_multiprocess_env and cfg.num_envs > 1:
            from catan_ai.env.vec_env import SubprocVecEnv
            self.vec_env = SubprocVecEnv(cfg.num_envs, env_config, cfg.seed)
            self.action_space_size = self.vec_env.action_space_size
            self.hex_feat_dim = self.vec_env.hex_feat_dim
            self.vertex_feat_dim = self.vec_env.vertex_feat_dim
            self.edge_feat_dim = self.vec_env.edge_feat_dim
            self.player_feat_dim = self.vec_env.player_feat_dim
            self.num_tiles = self.vec_env.num_tiles
            self.num_nodes = self.vec_env.num_nodes
            self.num_edges = self.vec_env.num_edges
            self._vertex_to_hex_adj = self.vec_env.vertex_to_hex_adj
            self._hex_to_vertex_adj = self.vec_env.hex_to_vertex_adj
            self._vertex_adj = self.vec_env.vertex_adj
            self._use_vec = True
        else:
            from catan_ai.env.catan_env import CatanTrainEnv
            self.envs = []
            for i in range(cfg.num_envs):
                env = CatanTrainEnv(env_config)
                self.envs.append(env)
            env0 = self.envs[0]
            self.action_space_size = env0.action_space_size
            self.hex_feat_dim = env0.hex_feat_dim
            self.vertex_feat_dim = env0.vertex_feat_dim
            self.edge_feat_dim = env0.edge_feat_dim
            self.player_feat_dim = env0.player_feat_dim
            self.num_tiles = env0.num_tiles
            self.num_nodes = env0.num_nodes
            self.num_edges = env0.num_edges
            self._vertex_to_hex_adj = env0.vertex_to_hex_adj
            self._hex_to_vertex_adj = env0.hex_to_vertex_adj
            self._vertex_adj = env0.vertex_adj
            self._use_vec = False

    def _np_obs_to_gpu(self, obs_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Transfer numpy observations to GPU efficiently.

        Uses pinned memory + non_blocking transfer on CUDA for overlap
        with CPU env stepping.
        """
        obs_gpu = {}
        if self.use_cuda and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for k, v in obs_np.items():
                    # Pin memory for async transfer
                    t = torch.from_numpy(v).float()
                    if t.is_pinned() is False:
                        t = t.pin_memory()
                    obs_gpu[k] = t.to(self.device, non_blocking=True)
            self._transfer_stream.synchronize()
        else:
            for k, v in obs_np.items():
                obs_gpu[k] = torch.from_numpy(v).float().to(self.device)
        return obs_gpu

    def _collect_rollout_vec(self):
        """Collect rollout using SubprocVecEnv (parallel CPU stepping)."""
        self.buffer.reset()
        cfg = self.config

        for step in range(cfg.num_steps):
            obs_gpu = self._np_obs_to_gpu(self._current_obs_np)

            with torch.no_grad():
                actions, log_probs, _, values = self.policy.get_action_and_value(
                    obs_gpu, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj
                )

            # Send actions to workers (runs in parallel while we could do other work)
            actions_np = actions.cpu().numpy()
            obs_np, rewards_np, dones_np, terminateds, infos = self.vec_env.step(actions_np)

            rewards = torch.from_numpy(rewards_np).float().to(self.device)
            dones = torch.from_numpy(dones_np).float().to(self.device)

            self.buffer.add(obs_gpu, actions, log_probs, rewards, dones, values)
            self._current_obs_np = obs_np

            # Track stats
            for i in range(cfg.num_envs):
                self.global_step += 1
                if dones_np[i]:
                    self.game_count += 1
                    if terminateds[i]:
                        if rewards_np[i] > 0:
                            self.win_count += 1
                        else:
                            self.loss_count += 1

        # Compute GAE
        with torch.no_grad():
            next_obs_gpu = self._np_obs_to_gpu(self._current_obs_np)
            next_value = self.policy.get_value(
                next_obs_gpu, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj
            )
        next_done = torch.zeros(cfg.num_envs, device=self.device)
        self.buffer.compute_gae(next_value, next_done, cfg.gamma, cfg.gae_lambda)

    def _collect_rollout_serial(self):
        """Collect rollout using in-process envs (fallback for single-process)."""
        self.buffer.reset()
        cfg = self.config

        for step in range(cfg.num_steps):
            obs_list = [self.envs[i]._current_obs for i in range(cfg.num_envs)]
            obs_np = {}
            for key in obs_list[0]:
                obs_np[key] = np.stack([obs_list[i][key] for i in range(cfg.num_envs)])
            obs_gpu = self._np_obs_to_gpu(obs_np)

            with torch.no_grad():
                actions, log_probs, _, values = self.policy.get_action_and_value(
                    obs_gpu, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj
                )

            rewards = torch.zeros(cfg.num_envs, device=self.device)
            dones = torch.zeros(cfg.num_envs, device=self.device)

            for i in range(cfg.num_envs):
                action_idx = actions[i].item()
                obs, reward, terminated, truncated, info = self.envs[i].step(action_idx)
                self.envs[i]._current_obs = obs
                rewards[i] = reward
                done = terminated or truncated
                dones[i] = float(done)
                if done:
                    self.game_count += 1
                    if terminated:
                        if reward > 0:
                            self.win_count += 1
                        else:
                            self.loss_count += 1
                    obs, _ = self.envs[i].reset()
                    self.envs[i]._current_obs = obs
                self.global_step += 1

            self.buffer.add(obs_gpu, actions, log_probs, rewards, dones, values)

        with torch.no_grad():
            obs_list = [self.envs[i]._current_obs for i in range(cfg.num_envs)]
            obs_np = {}
            for key in obs_list[0]:
                obs_np[key] = np.stack([obs_list[i][key] for i in range(cfg.num_envs)])
            next_obs_gpu = self._np_obs_to_gpu(obs_np)
            next_value = self.policy.get_value(
                next_obs_gpu, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj
            )
        next_done = torch.zeros(cfg.num_envs, device=self.device)
        self.buffer.compute_gae(next_value, next_done, cfg.gamma, cfg.gae_lambda)

    def _update_policy(self) -> Dict[str, float]:
        cfg = self.config

        frac = self.global_step / cfg.total_timesteps
        ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac

        if cfg.lr_schedule == "linear":
            lr = cfg.learning_rate * max(1.0 - frac, 0.0)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = cfg.learning_rate

        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        num_updates = 0

        for epoch in range(cfg.num_epochs):
            for mb in self.buffer.get_batches(cfg.num_minibatches):
                mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = mb

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    mb_obs, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj,
                    action=mb_actions,
                )

                if cfg.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss1 = (new_values - mb_returns) ** 2
                    v_loss2 = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                num_updates += 1

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        n = max(num_updates, 1)
        return {
            "policy_loss": total_pg_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "learning_rate": lr,
            "entropy_coef": ent_coef,
        }

    def train(self):
        cfg = self.config

        use_wandb = False
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=vars(cfg),
                name=f"catan-ppo-{cfg.seed}",
            )
            use_wandb = True
        except Exception:
            print("Wandb not available, logging to stdout only.")

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Initialize environments
        if self._use_vec:
            seeds = [cfg.seed + i for i in range(cfg.num_envs)]
            self._current_obs_np, _ = self.vec_env.reset(seeds)
        else:
            for i, env in enumerate(self.envs):
                obs, _ = env.reset(seed=cfg.seed + i)
                env._current_obs = obs

        num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
        batch_size = cfg.num_steps * cfg.num_envs
        mb_size = batch_size // cfg.num_minibatches

        print(f"Starting training: {num_updates} updates, {cfg.total_timesteps} total steps")
        print(f"Batch size: {batch_size}, minibatch: {mb_size}")
        print(f"Envs: {cfg.num_envs} ({'multiprocess' if self._use_vec else 'serial'})")
        print(f"Device: {self.device}")

        start_time = time.time()

        for update in range(1, num_updates + 1):
            t0 = time.time()

            # Collect rollout
            if self._use_vec:
                self._collect_rollout_vec()
            else:
                self._collect_rollout_serial()

            t_collect = time.time() - t0

            # PPO update
            t1 = time.time()
            stats = self._update_policy()
            t_update = time.time() - t1

            # Save to opponent pool
            if self.global_step % cfg.opponent_update_freq < cfg.num_steps * cfg.num_envs:
                self.opponent_pool.add(self.policy.state_dict())

            # Save checkpoint
            if self.global_step % cfg.save_freq < cfg.num_steps * cfg.num_envs:
                path = os.path.join(cfg.checkpoint_dir, f"policy_{self.global_step}.pt")
                torch.save({
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "config": vars(cfg),
                }, path)
                print(f"  Saved checkpoint: {path}")

            # Logging
            if update % cfg.log_freq == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                win_rate = self.win_count / max(self.game_count, 1)

                log_data = {
                    "global_step": self.global_step,
                    "sps": sps,
                    "win_rate": win_rate,
                    "games_played": self.game_count,
                    "opponent_pool_size": len(self.opponent_pool),
                    "time_collect": t_collect,
                    "time_update": t_update,
                    "gpu_util_ratio": t_update / (t_collect + t_update + 1e-8),
                    **stats,
                }

                if update % (cfg.log_freq * 10) == 0:
                    print(
                        f"Step {self.global_step:>8d} | "
                        f"SPS {sps:.0f} | "
                        f"WR {win_rate:.2%} ({self.game_count} games) | "
                        f"PL {stats['policy_loss']:.4f} | "
                        f"VL {stats['value_loss']:.4f} | "
                        f"Ent {stats['entropy']:.4f} | "
                        f"KL {stats['approx_kl']:.4f} | "
                        f"Collect {t_collect:.1f}s Update {t_update:.1f}s"
                    )

                if use_wandb:
                    wandb.log(log_data)

        # Final save
        path = os.path.join(cfg.checkpoint_dir, "policy_final.pt")
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": vars(cfg),
        }, path)
        print(f"Training complete. Final checkpoint: {path}")

        if self._use_vec:
            self.vec_env.close()

        if use_wandb:
            wandb.finish()

        return self.policy
