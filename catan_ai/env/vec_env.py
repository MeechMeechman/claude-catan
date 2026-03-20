"""
Vectorized environment using multiprocessing for parallel game stepping.

Each catanatron game runs in its own process, removing the CPU bottleneck
that would otherwise starve the GPU during rollout collection.
"""

import multiprocessing as mp
import numpy as np
from typing import Dict, List, Optional, Tuple


def _worker(
    pipe: mp.connection.Connection,
    env_config: dict,
    worker_id: int,
):
    """Worker process that runs a single CatanTrainEnv."""
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "catanatron_repo", "catanatron"))

    from catan_ai.env.catan_env import CatanTrainEnv
    env = CatanTrainEnv(env_config)

    while True:
        cmd, data = pipe.recv()
        if cmd == "reset":
            obs, info = env.reset(seed=data)
            pipe.send(("obs", obs, info))
        elif cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                next_obs, next_info = env.reset()
                pipe.send(("step", obs, reward, done, terminated, info, next_obs, next_info))
            else:
                pipe.send(("step", obs, reward, done, terminated, info, None, None))
        elif cmd == "get_spaces":
            pipe.send((
                env.action_space_size,
                env.hex_feat_dim,
                env.vertex_feat_dim,
                env.edge_feat_dim,
                env.player_feat_dim,
                env.num_tiles,
                env.num_nodes,
                env.num_edges,
                env.vertex_to_hex_adj.copy(),
                env.hex_to_vertex_adj.copy(),
                env.vertex_adj.copy(),
            ))
        elif cmd == "close":
            pipe.close()
            return


class SubprocVecEnv:
    """Vectorized env that runs CatanTrainEnv instances in subprocesses.

    This parallelizes the CPU-bound game stepping so the GPU isn't starved.
    """

    def __init__(self, num_envs: int, env_config: dict, base_seed: int = 42):
        self.num_envs = num_envs
        self.waiting = False

        ctx = mp.get_context("spawn")
        self.parent_pipes = []
        self.procs = []

        for i in range(num_envs):
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child_pipe, env_config, i),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self.parent_pipes.append(parent_pipe)
            self.procs.append(proc)

        # Get space info from first worker
        self.parent_pipes[0].send(("get_spaces", None))
        spaces_info = self.parent_pipes[0].recv()
        (
            self.action_space_size,
            self.hex_feat_dim,
            self.vertex_feat_dim,
            self.edge_feat_dim,
            self.player_feat_dim,
            self.num_tiles,
            self.num_nodes,
            self.num_edges,
            self.vertex_to_hex_adj,
            self.hex_to_vertex_adj,
            self.vertex_adj,
        ) = spaces_info

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[Dict[str, np.ndarray], List[dict]]:
        """Reset all envs and return stacked observations."""
        seeds = seeds or [None] * self.num_envs
        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("reset", seed))

        obs_list = []
        info_list = []
        for pipe in self.parent_pipes:
            _, obs, info = pipe.recv()
            obs_list.append(obs)
            info_list.append(info)

        return self._stack_obs(obs_list), info_list

    def step_async(self, actions: np.ndarray):
        """Send actions to all workers."""
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", int(action)))
        self.waiting = True

    def step_wait(self):
        """Wait for all workers and collect results."""
        results = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting = False

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, (_, obs, reward, done, terminated, info, next_obs, next_info) in enumerate(results):
            if done and next_obs is not None:
                obs_list.append(next_obs)  # auto-reset: return the new obs
                infos.append(next_info)
            else:
                obs_list.append(obs)
                infos.append(info)
            rewards[i] = reward
            dones[i] = float(done)
            terminateds[i] = terminated

        return self._stack_obs(obs_list), rewards, dones, terminateds, infos

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def _stack_obs(self, obs_list: List[Dict]) -> Dict[str, np.ndarray]:
        stacked = {}
        for key in obs_list[0]:
            stacked[key] = np.stack([obs[key] for obs in obs_list])
        return stacked

    def close(self):
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
