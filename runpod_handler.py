"""
RunPod serverless handler for Catan AI training and inference.

Supports two modes:
1. "train": Run full training pipeline (long-running pod)
2. "evaluate": Run evaluation against baselines
3. "infer": Get action for a given game state
"""

import os
import json
import torch

# Set paths before any other imports
os.environ.setdefault("PYTHONPATH", "/app:/app/catanatron_repo/catanatron")

import runpod


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a trained policy from checkpoint."""
    from catan_ai.env.catan_env import CatanTrainEnv
    from catan_ai.models.policy import CatanPolicy

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    return policy, env


def handler(event):
    """RunPod handler function."""
    job_input = event.get("input", {})
    mode = job_input.get("mode", "train")

    if mode == "train":
        return handle_train(job_input)
    elif mode == "evaluate":
        return handle_evaluate(job_input)
    else:
        return {"error": f"Unknown mode: {mode}"}


def handle_train(job_input: dict):
    """Run training pipeline."""
    from catan_ai.training.ppo import PPOConfig, PPOTrainer

    # Override config from job input
    config_overrides = job_input.get("config", {})
    config = PPOConfig(**{
        k: v for k, v in config_overrides.items()
        if hasattr(PPOConfig, k)
    })

    # Run imitation learning warmstart if requested
    if job_input.get("bc_warmstart", False):
        from catan_ai.training.imitation import generate_demonstrations, train_bc
        from catan_ai.env.catan_env import CatanTrainEnv
        from catan_ai.models.policy import CatanPolicy

        env = CatanTrainEnv({
            "num_players": config.num_players,
            "map_type": config.map_type,
        })

        policy = CatanPolicy(
            hex_input_dim=env.hex_feat_dim,
            vertex_input_dim=env.vertex_feat_dim,
            edge_input_dim=env.edge_feat_dim,
            player_input_dim=env.player_feat_dim,
            action_space_size=env.action_space_size,
            hidden_dim=config.hidden_dim,
            encoder_layers=config.encoder_layers,
            encoder_output_dim=config.encoder_output_dim,
        )

        num_demo_games = job_input.get("bc_num_games", 5000)
        demos = generate_demonstrations(
            num_games=num_demo_games,
            teacher=job_input.get("bc_teacher", "value"),
        )
        policy = train_bc(
            policy, demos, env,
            num_epochs=job_input.get("bc_epochs", 15),
            device=config.device,
            save_path="checkpoints/bc_warmstart.pt",
        )

    # PPO training
    trainer = PPOTrainer(config)

    # Load BC warmstart if available
    bc_path = "checkpoints/bc_warmstart.pt"
    if os.path.exists(bc_path):
        trainer.policy.load_state_dict(torch.load(bc_path, weights_only=True))
        print("Loaded BC warmstart weights")

    trainer.train()

    return {
        "status": "complete",
        "global_steps": trainer.global_step,
        "win_rate": trainer.win_count / max(trainer.game_count, 1),
        "games_played": trainer.game_count,
        "checkpoint": "checkpoints/policy_final.pt",
    }


def handle_evaluate(job_input: dict):
    """Run evaluation of a checkpoint."""
    from catan_ai.eval.tournament import Tournament

    checkpoint_path = job_input.get("checkpoint", "checkpoints/policy_final.pt")
    num_games = job_input.get("num_games", 500)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy, env = load_policy(checkpoint_path, device)

    tournament = Tournament(
        num_players=2,
        map_type="BASE",
    )

    results = tournament.evaluate_against_baselines(
        policy, env, num_games=num_games, device=device
    )
    tournament.print_summary()

    return {
        "results": {
            name: {k: v for k, v in data.items() if k != "ci_95"}
            for name, data in results.items()
        },
        "elo_ratings": tournament.elo_ratings,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
