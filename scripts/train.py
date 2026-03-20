#!/usr/bin/env python3
"""
Main training entrypoint for Catan AI.

Stages:
1. (Optional) Generate teacher demonstrations and BC warmstart
2. PPO self-play training with opponent diversity
3. Evaluation against baselines

Usage:
    python scripts/train.py                           # Full pipeline
    python scripts/train.py --skip-bc                 # Skip BC warmstart
    python scripts/train.py --total-steps 1000000     # Quick run
    python scripts/train.py --eval-only --checkpoint checkpoints/policy_final.pt
"""

import argparse
import os
import sys
import torch

# Ensure project is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "catanatron_repo", "catanatron"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Catan AI")

    # Pipeline control
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavioral cloning warmstart")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")

    # BC settings
    parser.add_argument("--bc-games", type=int, default=5000, help="Number of demo games for BC")
    parser.add_argument("--bc-teacher", type=str, default="value", choices=["value", "alphabeta", "random"])
    parser.add_argument("--bc-epochs", type=int, default=15)

    # PPO settings
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-type", type=str, default="shaped", choices=["sparse", "shaped"])

    # Game settings
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"])

    # Eval settings
    parser.add_argument("--eval-games", type=int, default=200)

    # Infra
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb-project", type=str, default="catan-ai")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # GPU performance
    parser.add_argument("--no-multiprocess", action="store_true", help="Disable multiprocess envs")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.x)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    from catan_ai.env.catan_env import CatanTrainEnv
    from catan_ai.models.policy import CatanPolicy

    # Create reference env for dimensions
    env = CatanTrainEnv({
        "num_players": args.num_players,
        "map_type": args.map_type,
        "vps_to_win": args.vps_to_win,
        "reward_type": args.reward_type,
    })

    # Create policy
    policy = CatanPolicy(
        hex_input_dim=env.hex_feat_dim,
        vertex_input_dim=env.vertex_feat_dim,
        edge_input_dim=env.edge_feat_dim,
        player_input_dim=env.player_feat_dim,
        action_space_size=env.action_space_size,
        num_tiles=env.num_tiles,
        num_nodes=env.num_nodes,
        num_edges=env.num_edges,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
    ).to(device)

    print(f"Policy parameters: {policy.count_parameters():,}")

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Eval-only mode
    if args.eval_only:
        from catan_ai.eval.tournament import Tournament
        policy.eval()
        tournament = Tournament(num_players=args.num_players, map_type=args.map_type)
        tournament.evaluate_against_baselines(
            policy, env, num_games=args.eval_games, device=device
        )
        tournament.print_summary()
        return

    # Stage 1: Behavioral Cloning warmstart
    if not args.skip_bc and not args.checkpoint:
        print("\n=== Stage 1: Behavioral Cloning ===")
        from catan_ai.training.imitation import generate_demonstrations, train_bc

        demos = generate_demonstrations(
            num_games=args.bc_games,
            num_players=args.num_players,
            map_type=args.map_type,
            vps_to_win=args.vps_to_win,
            teacher=args.bc_teacher,
            seed=args.seed,
        )

        bc_path = os.path.join(args.checkpoint_dir, "bc_warmstart.pt")
        policy = train_bc(
            policy, demos, env,
            num_epochs=args.bc_epochs,
            device=device,
            save_path=bc_path,
        )

    # Stage 2: PPO Self-play
    print("\n=== Stage 2: PPO Self-Play Training ===")
    from catan_ai.training.ppo import PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        total_timesteps=args.total_steps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        num_players=args.num_players,
        map_type=args.map_type,
        vps_to_win=args.vps_to_win,
        reward_type=args.reward_type,
        device=device,
        seed=args.seed,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        use_multiprocess_env=not args.no_multiprocess,
        compile_model=args.compile,
    )

    trainer = PPOTrainer(ppo_config)

    # Load BC warmstart weights into trainer's policy
    bc_path = os.path.join(args.checkpoint_dir, "bc_warmstart.pt")
    if os.path.exists(bc_path) and not args.skip_bc:
        trainer.policy.load_state_dict(torch.load(bc_path, map_location=device, weights_only=True))
        print("Loaded BC warmstart into PPO trainer")

    trained_policy = trainer.train()

    # Stage 3: Final evaluation
    print("\n=== Stage 3: Evaluation ===")
    from catan_ai.eval.tournament import Tournament

    trained_policy.eval()
    tournament = Tournament(num_players=args.num_players, map_type=args.map_type)
    tournament.evaluate_against_baselines(
        trained_policy, env, num_games=args.eval_games, device=device
    )
    tournament.print_summary()

    print("\nDone!")


if __name__ == "__main__":
    main()
