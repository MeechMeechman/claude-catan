"""
Evaluation and Elo rating tournament system for Catan agents.
"""

import os
import sys
import math
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch

_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "catanatron_repo", "catanatron")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def compute_elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
    """Compute Elo rating updates. score_a: 1=win, 0.5=draw, 0=loss."""
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - expected_b)
    return new_a, new_b


def wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for win rate."""
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denominator = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denominator
    return max(0, center - spread), min(1, center + spread)


class Tournament:
    """Round-robin tournament between agents with Elo tracking."""

    def __init__(
        self,
        num_players: int = 2,
        map_type: str = "BASE",
        vps_to_win: int = 10,
    ):
        self.num_players = num_players
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.results: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.elo_ratings: Dict[str, float] = {}

    def evaluate_against_baselines(
        self,
        policy: "CatanPolicy",
        env_ref: "CatanTrainEnv",
        num_games: int = 200,
        device: str = "cpu",
    ) -> Dict:
        """Evaluate a policy against standard baselines."""
        from catanatron.models.player import Color, RandomPlayer
        from catanatron.players.value import ValueFunctionPlayer

        baselines = {
            "random": RandomPlayer(Color.RED),
            "value_fn": ValueFunctionPlayer(Color.RED),
        }

        results = {}
        for name, baseline in baselines.items():
            wins, losses, draws = self._play_matches(
                policy, baseline, env_ref, num_games, device
            )
            total = wins + losses + draws
            win_rate = wins / max(total, 1)
            ci_low, ci_high = wilson_ci(wins, total)

            results[name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "ci_95": (ci_low, ci_high),
            }

            # Update Elo
            agent_name = "trained_policy"
            if agent_name not in self.elo_ratings:
                self.elo_ratings[agent_name] = 1200.0
            if name not in self.elo_ratings:
                self.elo_ratings[name] = 1000.0

            for _ in range(wins):
                self.elo_ratings[agent_name], self.elo_ratings[name] = compute_elo_update(
                    self.elo_ratings[agent_name], self.elo_ratings[name], 1.0
                )
            for _ in range(losses):
                self.elo_ratings[agent_name], self.elo_ratings[name] = compute_elo_update(
                    self.elo_ratings[agent_name], self.elo_ratings[name], 0.0
                )

            print(
                f"  vs {name:>12s}: {win_rate:.1%} "
                f"({wins}W/{losses}L/{draws}D) "
                f"[{ci_low:.1%}, {ci_high:.1%}] "
                f"Elo: {self.elo_ratings[agent_name]:.0f}"
            )

        return results

    def _play_matches(
        self,
        policy: "CatanPolicy",
        opponent,
        env_ref: "CatanTrainEnv",
        num_games: int,
        device: str,
    ) -> Tuple[int, int, int]:
        """Play matches and return (wins, losses, draws)."""
        from catan_ai.env.catan_env import CatanTrainEnv
        from catan_ai.models.policy import PolicyPlayer

        wins, losses, draws = 0, 0, 0

        env = CatanTrainEnv({
            "num_players": self.num_players,
            "map_type": self.map_type,
            "vps_to_win": self.vps_to_win,
            "reward_type": "sparse",
        })
        env.enemies = [opponent]
        env.players = [env.p0] + [opponent]
        env.player_colors = tuple(p.color for p in env.players)

        for game_idx in range(num_games):
            obs, info = env.reset(seed=game_idx * 31)
            valid_actions = info.get("valid_actions", [])

            for step in range(1000):
                if not valid_actions:
                    break

                # Get policy action
                obs_t = {
                    k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                    for k, v in obs.items()
                }

                v2h = torch.tensor(env.vertex_to_hex_adj, dtype=torch.long, device=device)
                h2v = torch.tensor(env.hex_to_vertex_adj, dtype=torch.long, device=device)
                va = torch.tensor(env.vertex_adj, dtype=torch.long, device=device)

                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(
                        obs_t, v2h, h2v, va, deterministic=True
                    )

                action_idx = action.item()

                # Validate action
                if action_idx not in valid_actions:
                    action_idx = random.choice(valid_actions) if valid_actions else 0

                obs, reward, terminated, truncated, info = env.step(action_idx)
                valid_actions = info.get("valid_actions", [])

                if terminated:
                    if reward > 0:
                        wins += 1
                    else:
                        losses += 1
                    break
                if truncated:
                    draws += 1
                    break
            else:
                draws += 1

        return wins, losses, draws

    def print_summary(self):
        print("\n=== Tournament Summary ===")
        print(f"{'Agent':<20s} {'Elo':>6s}")
        print("-" * 28)
        for name, elo in sorted(self.elo_ratings.items(), key=lambda x: -x[1]):
            print(f"{name:<20s} {elo:>6.0f}")
