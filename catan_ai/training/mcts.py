"""
AlphaZero-style MCTS for Catan.

GPU efficiency strategy for A5000 (24GB):
- Batched leaf evaluation: collect leaves from N parallel games,
  evaluate them in one GPU forward pass
- Async CPU game simulation: game copies + opponent moves run on CPU
  while GPU processes the neural net batch
- Replay buffer in CPU RAM, sampled in large batches for training
- Separate self-play and training phases to maximize GPU utilization

Training loop:
1. Self-play: Run N parallel games with MCTS (CPU game sim + GPU eval)
2. Store (state, mcts_policy, outcome) in replay buffer
3. Train network on mini-batches from replay buffer
4. Repeat
"""

import math
import os
import copy
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from catan_ai.models.policy import CatanPolicy


@dataclass
class MCTSConfig:
    # MCTS search
    num_simulations: int = 64         # simulations per move
    c_puct: float = 1.5               # exploration constant
    dirichlet_alpha: float = 0.3      # root noise for exploration
    dirichlet_epsilon: float = 0.25   # fraction of noise at root
    temperature: float = 1.0          # action selection temperature (early game)
    temp_threshold: int = 30          # switch to greedy after this many moves

    # Self-play
    num_parallel_games: int = 32      # games running simultaneously
    max_moves_per_game: int = 500

    # Training
    total_iterations: int = 100       # AlphaZero iterations
    games_per_iteration: int = 256    # self-play games per iteration
    training_epochs: int = 10
    training_batch_size: int = 512
    replay_buffer_size: int = 200_000
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"       # cosine | constant
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # Environment
    num_players: int = 2
    map_type: str = "BASE"
    vps_to_win: int = 10

    # Model
    hidden_dim: int = 256
    encoder_layers: int = 6
    encoder_output_dim: int = 256

    # Eval
    eval_games: int = 100
    eval_simulations: int = 64        # MCTS sims during eval

    # Infra
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_freq: int = 1

    # Opponent for eval
    eval_vs_value_fn: bool = True


# ---- MCTS Core ----

class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = ['visit_count', 'value_sum', 'prior', 'children', 'is_terminal']

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_terminal = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0


def select_child(node: MCTSNode, c_puct: float) -> Tuple[int, 'MCTSNode']:
    """Select best child using PUCT formula (AlphaZero variant)."""
    best_score = -float('inf')
    best_action = -1
    best_child = None
    sqrt_total = math.sqrt(node.visit_count + 1)

    for action, child in node.children.items():
        q = child.value()
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child


def expand_node(node: MCTSNode, policy_probs: np.ndarray, action_mask: np.ndarray):
    """Expand a node with policy priors, only for legal actions."""
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) == 0:
        return

    # Renormalize policy over legal actions
    legal_probs = policy_probs[legal_actions]
    prob_sum = legal_probs.sum()
    if prob_sum > 0:
        legal_probs = legal_probs / prob_sum
    else:
        legal_probs = np.ones(len(legal_actions)) / len(legal_actions)

    for i, action in enumerate(legal_actions):
        node.children[action] = MCTSNode(prior=legal_probs[i])


def add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
    """Add Dirichlet noise to root node priors for exploration."""
    actions = list(node.children.keys())
    if not actions:
        return
    noise = np.random.dirichlet([alpha] * len(actions))
    for i, action in enumerate(actions):
        node.children[action].prior = (
            (1 - epsilon) * node.children[action].prior + epsilon * noise[i]
        )


def get_mcts_policy(node: MCTSNode, action_space_size: int, temperature: float) -> np.ndarray:
    """Extract improved policy from visit counts."""
    visits = np.zeros(action_space_size, dtype=np.float32)
    for action, child in node.children.items():
        visits[action] = child.visit_count

    if temperature == 0:
        # Greedy
        policy = np.zeros_like(visits)
        policy[visits.argmax()] = 1.0
        return policy

    # Temperature-scaled
    if visits.sum() == 0:
        return visits
    visits = visits ** (1.0 / temperature)
    total = visits.sum()
    if total > 0:
        return visits / total
    return visits


# ---- Game Simulation for MCTS ----

class GameSimulator:
    """Wraps a catanatron game for MCTS simulation.

    Each simulator holds a game state that can be copied and advanced.
    Observations are extracted for neural network evaluation.
    """

    def __init__(self, env_config: dict):
        import sys
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "catanatron_repo", "catanatron"))

        from catan_ai.env.catan_env import CatanTrainEnv
        self.env = CatanTrainEnv(env_config)
        self.obs = None
        self.info = None
        self.done = False
        self.reward = 0.0

    def reset(self, seed=None):
        self.obs, self.info = self.env.reset(seed=seed)
        self.done = False
        self.reward = 0.0
        return self.obs

    def get_action_mask(self) -> np.ndarray:
        return self.obs["action_mask"]

    def step(self, action: int):
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated
        return self.obs, self.reward, self.done

    def copy_game_state(self):
        """Return a deep copy of the current game for tree simulation."""
        return copy.deepcopy(self.env.game)

    def simulate_action(self, game_copy, action: int) -> Tuple[dict, float, bool]:
        """Execute an action on a game copy and return new obs + reward + done.

        This is the core MCTS simulation step — it advances the copied game
        state without modifying the real environment.
        """
        from catanatron.gym.envs.action_space import from_action_space
        from catanatron.game import TURNS_LIMIT

        catan_action = from_action_space(
            action, self.env.p0_color, self.env.player_colors, self.env.map_type
        )

        if catan_action in game_copy.playable_actions:
            game_copy.execute(catan_action)
        else:
            # Invalid action — pick first valid
            if game_copy.playable_actions:
                game_copy.execute(game_copy.playable_actions[0])

        # Advance opponent turns
        while (
            game_copy.winning_color() is None
            and game_copy.state.num_turns < TURNS_LIMIT
            and game_copy.state.current_color() != self.env.p0_color
        ):
            game_copy.play_tick()

        # Check terminal
        winning_color = game_copy.winning_color()
        terminated = winning_color is not None
        truncated = game_copy.state.num_turns >= TURNS_LIMIT

        if terminated:
            reward = 1.0 if winning_color == self.env.p0_color else -1.0
        elif truncated:
            reward = 0.0
        else:
            reward = 0.0

        done = terminated or truncated

        # Extract observation from the copied game
        old_game = self.env.game
        self.env.game = game_copy
        obs = self.env._get_obs()
        self.env.game = old_game

        return obs, reward, done


# ---- Batched MCTS with GPU Evaluation ----

class BatchedMCTS:
    """Runs MCTS across multiple games with batched neural network evaluation.

    Key GPU optimization: instead of evaluating one leaf at a time, we collect
    all leaves that need evaluation across all parallel games, batch them into
    one GPU forward pass, then distribute results back.
    """

    def __init__(
        self,
        policy: CatanPolicy,
        device: torch.device,
        adj_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        config: MCTSConfig,
    ):
        self.policy = policy
        self.device = device
        self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj = adj_tensors
        self.config = config

    @torch.no_grad()
    def _evaluate_batch(self, obs_list: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of observations on GPU.

        Args:
            obs_list: list of observation dicts (numpy arrays)
        Returns:
            policies: (N, action_space_size) numpy
            values: (N,) numpy
        """
        if not obs_list:
            return np.array([]), np.array([])

        # Stack observations
        obs_batch = {}
        for key in obs_list[0]:
            obs_batch[key] = torch.from_numpy(
                np.stack([obs[key] for obs in obs_list])
            ).float().to(self.device)

        logits, values = self.policy.forward(
            obs_batch, self.vertex_to_hex, self.hex_to_vertex, self.vertex_adj
        )

        # Convert logits to probabilities (with masking already applied)
        policies = F.softmax(logits, dim=-1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()

        return policies, values

    def search(
        self,
        simulators: List[GameSimulator],
        num_simulations: int,
    ) -> List[Tuple[np.ndarray, MCTSNode]]:
        """Run MCTS for all simulators, return (mcts_policy, root_node) per game.

        This batches leaf evaluations across all games for GPU efficiency.
        """
        cfg = self.config
        action_space_size = simulators[0].env.action_space_size
        num_games = len(simulators)

        # Initialize root nodes
        roots = [MCTSNode() for _ in range(num_games)]

        # Evaluate roots in batch
        root_obs = [sim.obs for sim in simulators]
        root_policies, root_values = self._evaluate_batch(root_obs)

        for i in range(num_games):
            mask = simulators[i].get_action_mask()
            expand_node(roots[i], root_policies[i], mask)
            add_dirichlet_noise(roots[i], cfg.dirichlet_alpha, cfg.dirichlet_epsilon)

        # Run simulations
        for sim_idx in range(num_simulations):
            # For each game, traverse tree and collect leaves needing evaluation
            leaves_to_eval = []  # (game_idx, obs, search_path, game_copy)
            backup_ready = []    # (game_idx, search_path, value) — terminal nodes

            for g in range(num_games):
                node = roots[g]
                search_path = [node]
                game_copy = simulators[g].copy_game_state()
                hit_terminal = False

                # SELECT: walk down the tree
                while node.expanded() and not node.is_terminal:
                    action, child = select_child(node, cfg.c_puct)
                    node = child
                    search_path.append(node)

                    # Simulate this action on the game copy
                    _, reward, done = simulators[g].simulate_action(game_copy, action)

                    if done:
                        node.is_terminal = True
                        backup_ready.append((g, search_path, reward))
                        hit_terminal = True
                        break

                if not hit_terminal and not node.is_terminal and not node.expanded():
                    # Extract obs from the game copy for neural net evaluation
                    old_game = simulators[g].env.game
                    simulators[g].env.game = game_copy
                    leaf_obs = simulators[g].env._get_obs()
                    simulators[g].env.game = old_game

                    leaves_to_eval.append((g, leaf_obs, search_path, game_copy))

            # EVALUATE: batch all leaves on GPU
            if leaves_to_eval:
                leaf_obs_list = [item[1] for item in leaves_to_eval]
                policies, values = self._evaluate_batch(leaf_obs_list)

                for idx, (g, leaf_obs, search_path, game_copy) in enumerate(leaves_to_eval):
                    mask = leaf_obs["action_mask"]
                    leaf_node = search_path[-1]
                    expand_node(leaf_node, policies[idx], mask)
                    backup_ready.append((g, search_path, values[idx]))

            # BACKPROPAGATE
            for g, search_path, value in backup_ready:
                for node in reversed(search_path):
                    node.visit_count += 1
                    node.value_sum += value

        # Extract policies from visit counts
        results = []
        for g in range(num_games):
            move_num = getattr(simulators[g], '_move_count', 0)
            temp = cfg.temperature if move_num < cfg.temp_threshold else 0.0
            mcts_policy = get_mcts_policy(roots[g], action_space_size, temp)
            results.append((mcts_policy, roots[g]))

        return results


# ---- Replay Buffer ----

class ReplayBuffer:
    """Fixed-size circular buffer for AlphaZero training samples."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs: dict, mcts_policy: np.ndarray, outcome: float):
        self.buffer.append((
            {k: v.copy() for k, v in obs.items()},
            mcts_policy.copy(),
            outcome,
        ))

    def sample(self, batch_size: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]

        obs_batch = {}
        for key in batch[0][0]:
            obs_batch[key] = np.stack([b[0][key] for b in batch])

        policies = np.stack([b[1] for b in batch])
        outcomes = np.array([b[2] for b in batch], dtype=np.float32)

        return obs_batch, policies, outcomes

    def __len__(self):
        return len(self.buffer)


# ---- AlphaZero Trainer ----

class AlphaZeroTrainer:
    """AlphaZero-style training loop.

    GPU efficiency on A5000:
    - Self-play: batched MCTS eval across parallel games (~32 games)
      Each simulation collects up to 32 leaves and evaluates in one forward pass
    - Training: large batches (512) from replay buffer, standard supervised learning
    - Memory: replay buffer in CPU RAM (~200K samples ≈ 4GB), only batches on GPU
    """

    def __init__(self, config: MCTSConfig):
        self.config = config
        self.device = torch.device(config.device)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create reference env for dimensions
        self.env_config = {
            "num_players": config.num_players,
            "map_type": config.map_type,
            "vps_to_win": config.vps_to_win,
            "reward_type": "sparse",  # MCTS uses game outcomes, not shaped rewards
        }

        ref_sim = GameSimulator(self.env_config)
        ref_env = ref_sim.env

        # Create policy
        self.policy = CatanPolicy(
            hex_input_dim=ref_env.hex_feat_dim,
            vertex_input_dim=ref_env.vertex_feat_dim,
            edge_input_dim=ref_env.edge_feat_dim,
            player_input_dim=ref_env.player_feat_dim,
            action_space_size=ref_env.action_space_size,
            num_tiles=ref_env.num_tiles,
            num_nodes=ref_env.num_nodes,
            num_edges=ref_env.num_edges,
            hidden_dim=config.hidden_dim,
            encoder_layers=config.encoder_layers,
            encoder_output_dim=config.encoder_output_dim,
        ).to(self.device)

        print(f"MCTS Policy parameters: {self.policy.count_parameters():,}")

        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Adjacency tensors
        self.adj_tensors = (
            torch.tensor(ref_env.vertex_to_hex_adj, dtype=torch.long, device=self.device),
            torch.tensor(ref_env.hex_to_vertex_adj, dtype=torch.long, device=self.device),
            torch.tensor(ref_env.vertex_adj, dtype=torch.long, device=self.device),
        )

        self.action_space_size = ref_env.action_space_size
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # Create MCTS searcher
        self.mcts = BatchedMCTS(self.policy, self.device, self.adj_tensors, config)

    def self_play_batch(self) -> List[List[Tuple[dict, np.ndarray, float]]]:
        """Run a batch of self-play games using MCTS.

        Returns list of game trajectories, each a list of (obs, mcts_policy, outcome).
        """
        cfg = self.config
        self.policy.eval()

        num_games = cfg.num_parallel_games
        simulators = [GameSimulator(self.env_config) for _ in range(num_games)]

        # Reset all games
        for i, sim in enumerate(simulators):
            sim.reset(seed=cfg.seed + random.randint(0, 100000))
            sim._move_count = 0

        # Trajectories: per-game list of (obs, mcts_policy)
        trajectories = [[] for _ in range(num_games)]
        active = list(range(num_games))
        completed = []

        while active:
            # Get active simulators
            active_sims = [simulators[g] for g in active]

            # Run MCTS for all active games
            results = self.mcts.search(active_sims, cfg.num_simulations)

            # Select actions and step environments
            new_active = []
            for idx, g in enumerate(active):
                mcts_policy, root = results[idx]

                # Store trajectory step
                trajectories[g].append((
                    {k: v.copy() for k, v in simulators[g].obs.items()},
                    mcts_policy,
                ))

                # Sample action from MCTS policy
                action = np.random.choice(len(mcts_policy), p=mcts_policy)
                obs, reward, done = simulators[g].step(action)
                simulators[g]._move_count += 1

                if done or simulators[g]._move_count >= cfg.max_moves_per_game:
                    # Game over — assign outcome to all positions
                    outcome = reward  # +1 win, -1 loss, 0 draw/truncated
                    game_data = []
                    for obs_t, policy_t in trajectories[g]:
                        game_data.append((obs_t, policy_t, outcome))
                    completed.append(game_data)
                else:
                    new_active.append(g)

            active = new_active

        return completed

    def train_on_buffer(self) -> Dict[str, float]:
        """Train the network on samples from the replay buffer."""
        cfg = self.config
        self.policy.train()

        if len(self.replay_buffer) < cfg.training_batch_size:
            return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for epoch in range(cfg.training_epochs):
            obs_np, policies_np, outcomes_np = self.replay_buffer.sample(cfg.training_batch_size)

            # Move to GPU
            obs_gpu = {
                k: torch.from_numpy(v).float().to(self.device)
                for k, v in obs_np.items()
            }
            target_policies = torch.from_numpy(policies_np).float().to(self.device)
            target_values = torch.from_numpy(outcomes_np).float().to(self.device)

            # Forward
            logits, values = self.policy.forward(
                obs_gpu, *self.adj_tensors
            )
            values = values.squeeze(-1)

            # Policy loss: cross-entropy with MCTS policy
            # Clamp logits to avoid NaN from -inf in masked positions
            safe_logits = logits.clamp(min=-30.0)
            log_probs = F.log_softmax(safe_logits, dim=-1)
            policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

            # Value loss: MSE with game outcome
            value_loss = F.mse_loss(values, target_values)

            loss = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "total_loss": (total_policy_loss + total_value_loss) / n,
        }

    def evaluate(self, num_games: int = 100) -> Dict[str, float]:
        """Evaluate against baselines using MCTS for move selection."""
        self.policy.eval()
        from catan_ai.eval.tournament import Tournament

        ref_sim = GameSimulator(self.env_config)
        tournament = Tournament(
            num_players=self.config.num_players,
            map_type=self.config.map_type,
        )
        tournament.evaluate_against_baselines(
            self.policy, ref_sim.env,
            num_games=num_games,
            device=str(self.device),
        )
        tournament.print_summary()
        return tournament.results if hasattr(tournament, 'results') else {}

    def train(self):
        """Main AlphaZero training loop."""
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Optional wandb
        use_wandb = False
        try:
            import wandb
            wandb.init(
                project="catan-ai-mcts",
                config=vars(cfg),
                name=f"mcts-{cfg.seed}",
            )
            use_wandb = True
        except Exception:
            print("Wandb not available, logging to stdout only.")

        print(f"Starting AlphaZero training:")
        print(f"  Iterations: {cfg.total_iterations}")
        print(f"  Games/iteration: {cfg.games_per_iteration}")
        print(f"  Simulations/move: {cfg.num_simulations}")
        print(f"  Parallel games: {cfg.num_parallel_games}")
        print(f"  Replay buffer: {cfg.replay_buffer_size}")
        print(f"  Device: {self.device}")
        print()

        total_games = 0
        start_time = time.time()

        for iteration in range(1, cfg.total_iterations + 1):
            iter_start = time.time()

            # ---- Self-play phase ----
            sp_start = time.time()
            games_this_iter = 0
            wins = 0
            total_moves = 0

            while games_this_iter < cfg.games_per_iteration:
                game_trajectories = self.self_play_batch()

                for game_data in game_trajectories:
                    for obs, policy, outcome in game_data:
                        self.replay_buffer.add(obs, policy, outcome)
                    games_this_iter += 1
                    total_games += 1
                    total_moves += len(game_data)
                    if game_data and game_data[-1][2] > 0:
                        wins += 1

            sp_time = time.time() - sp_start
            win_rate = wins / max(games_this_iter, 1)
            avg_game_len = total_moves / max(games_this_iter, 1)

            # ---- Training phase ----
            tr_start = time.time()
            train_stats = self.train_on_buffer()
            tr_time = time.time() - tr_start

            # ---- Learning rate schedule ----
            if cfg.lr_schedule == "cosine":
                frac = iteration / cfg.total_iterations
                lr = cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * frac))
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
            else:
                lr = cfg.learning_rate

            # ---- Logging ----
            elapsed = time.time() - start_time
            games_per_sec = total_games / elapsed

            log_data = {
                "iteration": iteration,
                "total_games": total_games,
                "self_play_win_rate": win_rate,
                "avg_game_length": avg_game_len,
                "replay_buffer_size": len(self.replay_buffer),
                "self_play_time": sp_time,
                "training_time": tr_time,
                "games_per_sec": games_per_sec,
                "learning_rate": lr,
                **train_stats,
            }

            if iteration % cfg.log_freq == 0:
                print(
                    f"Iter {iteration:>4d}/{cfg.total_iterations} | "
                    f"Games {total_games:>6d} | "
                    f"WR {win_rate:.1%} | "
                    f"AvgLen {avg_game_len:.0f} | "
                    f"Buffer {len(self.replay_buffer):>6d} | "
                    f"PL {train_stats['policy_loss']:.4f} | "
                    f"VL {train_stats['value_loss']:.4f} | "
                    f"SP {sp_time:.1f}s TR {tr_time:.1f}s | "
                    f"G/s {games_per_sec:.2f}"
                )

            if use_wandb:
                wandb.log(log_data)

            # ---- Checkpoint ----
            if iteration % 10 == 0:
                path = os.path.join(cfg.checkpoint_dir, f"mcts_iter_{iteration}.pt")
                torch.save({
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "iteration": iteration,
                    "total_games": total_games,
                    "config": vars(cfg),
                }, path)
                print(f"  Saved: {path}")

            # ---- Periodic evaluation ----
            if iteration % 20 == 0 and cfg.eval_vs_value_fn:
                print(f"\n  === Evaluation at iteration {iteration} ===")
                self.evaluate(num_games=cfg.eval_games)
                print()

        # Final save
        path = os.path.join(cfg.checkpoint_dir, "mcts_final.pt")
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": cfg.total_iterations,
            "total_games": total_games,
            "config": vars(cfg),
        }, path)
        print(f"Training complete. Final: {path}")

        # Final evaluation
        print("\n=== Final Evaluation ===")
        self.evaluate(num_games=cfg.eval_games)

        if use_wandb:
            wandb.finish()

        return self.policy
