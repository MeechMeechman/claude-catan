#!/usr/bin/env python3
"""
Evaluate a trained Catan AI checkpoint against baselines.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt
    python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt --num-games 1000
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "catanatron_repo", "catanatron"))

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    from catan_ai.env.catan_env import CatanTrainEnv
    from catan_ai.models.policy import CatanPolicy
    from catan_ai.eval.tournament import Tournament

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    env = CatanTrainEnv({
        "num_players": config.get("num_players", 2),
        "map_type": config.get("map_type", "BASE"),
    })

    policy = CatanPolicy(
        hex_input_dim=env.hex_feat_dim,
        vertex_input_dim=env.vertex_feat_dim,
        edge_input_dim=env.edge_feat_dim,
        player_input_dim=env.player_feat_dim,
        action_space_size=env.action_space_size,
        num_tiles=env.num_tiles,
        num_nodes=env.num_nodes,
        num_edges=env.num_edges,
        hidden_dim=config.get("hidden_dim", 128),
        encoder_layers=config.get("encoder_layers", 4),
        encoder_output_dim=config.get("encoder_output_dim", 256),
    ).to(device)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    print(f"Loaded: {args.checkpoint} (step {checkpoint.get('global_step', '?')})")
    print(f"Parameters: {policy.count_parameters():,}")

    tournament = Tournament(
        num_players=config.get("num_players", 2),
        map_type=config.get("map_type", "BASE"),
    )

    print(f"\nEvaluating over {args.num_games} games per baseline...")
    tournament.evaluate_against_baselines(
        policy, env, num_games=args.num_games, device=device
    )
    tournament.print_summary()


if __name__ == "__main__":
    main()
