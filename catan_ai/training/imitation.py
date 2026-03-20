"""
Behavioral cloning (imitation learning) for warmstarting the policy.

Generates teacher demonstrations from catanatron's heuristic players,
then trains the policy via supervised cross-entropy loss on actions.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from catan_ai.env.catan_env import CatanTrainEnv
from catan_ai.models.policy import CatanPolicy


def generate_demonstrations(
    num_games: int = 10000,
    num_players: int = 2,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    teacher: str = "alphabeta",
    seed: int = 42,
    max_steps_per_game: int = 1000,
) -> List[Dict]:
    """Generate demonstration data from a heuristic teacher.

    Each demo is a dict with observation tensors and the teacher's action index.
    """
    import sys
    _REPO = os.path.join(os.path.dirname(__file__), "..", "..", "catanatron_repo", "catanatron")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    from catanatron.models.player import Color, RandomPlayer
    from catanatron.players.value import ValueFunctionPlayer

    # Create teacher player
    if teacher == "alphabeta":
        from catanatron.players.minimax import AlphaBetaPlayer
        enemy = AlphaBetaPlayer(Color.RED, depth=2, prunning=True)
    elif teacher == "value":
        enemy = ValueFunctionPlayer(Color.RED)
    else:
        enemy = RandomPlayer(Color.RED)

    env = CatanTrainEnv({
        "num_players": num_players,
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "reward_type": "sparse",
    })

    # Replace enemy with teacher
    env.enemies = [enemy]
    env.players = [env.p0] + [enemy]
    env.player_colors = tuple(p.color for p in env.players)

    demos = []
    random.seed(seed)

    for game_idx in tqdm(range(num_games), desc="Generating demos"):
        obs, info = env.reset(seed=seed + game_idx)
        valid_actions = info["valid_actions"]

        for step in range(max_steps_per_game):
            if not valid_actions:
                break

            # Teacher picks action using value function heuristic on P0's behalf
            # We use the env's game state to get the best heuristic action
            game = env.game

            # Evaluate each valid action using value function
            best_action_idx = _teacher_select(game, valid_actions, env)

            demos.append({
                "obs": {k: v.copy() for k, v in obs.items()},
                "action": best_action_idx,
            })

            obs, reward, terminated, truncated, info = env.step(best_action_idx)
            valid_actions = info.get("valid_actions", [])

            if terminated or truncated:
                break

    print(f"Generated {len(demos)} demonstration steps from {num_games} games")
    return demos


def _teacher_select(game, valid_actions, env):
    """Use value function to select best action from valid set."""
    from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
    from catanatron.gym.envs.action_space import from_action_space

    value_fn = base_fn(DEFAULT_WEIGHTS)
    best_val = float("-inf")
    best_action_idx = valid_actions[0]

    for action_idx in valid_actions:
        catan_action = from_action_space(
            action_idx, env.p0_color, env.player_colors, env.map_type
        )
        if catan_action not in game.playable_actions:
            continue

        game_copy = game.copy()
        game_copy.execute(catan_action)
        val = value_fn(game_copy, env.p0_color)

        if val > best_val:
            best_val = val
            best_action_idx = action_idx

    return best_action_idx


def train_bc(
    policy: CatanPolicy,
    demos: List[Dict],
    env: CatanTrainEnv,
    num_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> CatanPolicy:
    """Train policy via behavioral cloning on demonstrations."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Static adjacency
    v2h = torch.tensor(env.vertex_to_hex_adj, dtype=torch.long, device=device)
    h2v = torch.tensor(env.hex_to_vertex_adj, dtype=torch.long, device=device)
    vadj = torch.tensor(env.vertex_adj, dtype=torch.long, device=device)

    for epoch in range(num_epochs):
        random.shuffle(demos)
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(demos), batch_size):
            batch = demos[i:i + batch_size]

            # Collate batch
            obs_batch = {}
            for key in batch[0]["obs"]:
                obs_batch[key] = torch.stack([
                    torch.tensor(d["obs"][key], dtype=torch.float32)
                    for d in batch
                ]).to(device)

            actions = torch.tensor([d["action"] for d in batch], dtype=torch.long, device=device)

            # Forward
            logits, _ = policy.forward(obs_batch, v2h, h2v, vadj)
            loss = criterion(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += len(batch)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        print(f"BC Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3%}")

    if save_path:
        torch.save(policy.state_dict(), save_path)
        print(f"Saved BC policy to {save_path}")

    return policy
