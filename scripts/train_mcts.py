#!/usr/bin/env python3
"""
AlphaZero-style MCTS training for Catan AI.

Usage:
    python scripts/train_mcts.py                          # Default config
    python scripts/train_mcts.py --iterations 50 --sims 32  # Quick run
    python scripts/train_mcts.py --checkpoint checkpoints/mcts_iter_50.pt  # Resume

A5000 recommended settings (24GB VRAM):
    python scripts/train_mcts.py \
        --iterations 100 \
        --games-per-iter 256 \
        --sims 64 \
        --parallel-games 32 \
        --hidden-dim 256 \
        --encoder-layers 6 \
        --batch-size 512 \
        --buffer-size 200000
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "catanatron_repo", "catanatron"))


def parse_args():
    p = argparse.ArgumentParser(description="Train Catan AI with AlphaZero MCTS")

    # MCTS search
    p.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp-threshold", type=int, default=30)

    # Self-play
    p.add_argument("--parallel-games", type=int, default=32)
    p.add_argument("--games-per-iter", type=int, default=256)

    # Training
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Model
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--encoder-layers", type=int, default=6)

    # Eval
    p.add_argument("--eval-games", type=int, default=100)

    # Infra
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    from catan_ai.training.mcts import MCTSConfig, AlphaZeroTrainer

    config = MCTSConfig(
        num_simulations=args.sims,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temp_threshold=args.temp_threshold,
        num_parallel_games=args.parallel_games,
        games_per_iteration=args.games_per_iter,
        total_iterations=args.iterations,
        training_epochs=args.epochs,
        training_batch_size=args.batch_size,
        replay_buffer_size=args.buffer_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        eval_games=args.eval_games,
        device=device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = AlphaZeroTrainer(config)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        trainer.policy.load_state_dict(ckpt["policy_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    trainer.train()


if __name__ == "__main__":
    main()
