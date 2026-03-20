#!/bin/bash
# =============================================================
# RunPod RTX A5000 Setup & Training Script
# =============================================================
# Usage: Copy-paste this entire script into your RunPod terminal,
# OR: bash scripts/runpod_setup.sh
# =============================================================
set -e

echo "========================================"
echo " Catan AI - RunPod Setup"
echo "========================================"

# --- 1. System check ---
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# --- 2. Clone / update repo ---
echo ""
echo "[2/6] Setting up repository..."
cd /workspace

if [ -d "claude_catan" ]; then
    echo "  Repo exists, pulling latest..."
    cd claude_catan
    git pull --ff-only 2>/dev/null || echo "  (local changes, skipping pull)"
else
    echo "  Cloning fresh..."
    # Replace with your actual repo URL
    git clone https://github.com/<YOUR_USER>/claude_catan.git
    cd claude_catan
fi

# --- 3. Install dependencies ---
echo ""
echo "[3/6] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -e .
pip install -q wandb

# Install catanatron from submodule/clone
if [ ! -d "catanatron_repo" ]; then
    git clone --depth 1 https://github.com/bcollazo/catanatron.git catanatron_repo
fi
pip install -q -e catanatron_repo/catanatron

echo "  Dependencies installed."

# --- 4. Set environment ---
echo ""
echo "[4/6] Configuring environment..."
export PYTHONPATH="/workspace/claude_catan:/workspace/claude_catan/catanatron_repo/catanatron:$PYTHONPATH"

# Optional: set wandb key for cloud logging
# export WANDB_API_KEY="your-key-here"
# For offline logging (default):
export WANDB_MODE=offline

mkdir -p checkpoints logs demos

# --- 5. Quick smoke test ---
echo ""
echo "[5/6] Running smoke test..."
python3 -c "
import torch
from catan_ai.training.ppo import PPOConfig, PPOTrainer
config = PPOConfig(
    total_timesteps=512,
    num_envs=2,
    num_steps=64,
    num_epochs=1,
    num_minibatches=2,
    device='cuda',
    use_multiprocess_env=False,
)
trainer = PPOTrainer(config)
print('  Smoke test PASSED - GPU is working')
del trainer
torch.cuda.empty_cache()
"

echo ""
echo "[6/6] Setup complete!"
echo ""
echo "========================================"
echo " Ready to train. Recommended commands:"
echo "========================================"
echo ""
echo "  # Full pipeline (BC warmstart + PPO + eval):"
echo "  python3 scripts/train.py --num-envs 32 --num-steps 512 --total-steps 10000000"
echo ""
echo "  # Skip BC, PPO only:"
echo "  python3 scripts/train.py --skip-bc --num-envs 32 --total-steps 5000000"
echo ""
echo "  # Quick test run (5 min):"
echo "  python3 scripts/train.py --skip-bc --num-envs 16 --total-steps 100000"
echo ""
echo "  # Resume from checkpoint:"
echo "  python3 scripts/train.py --skip-bc --checkpoint checkpoints/policy_final.pt"
echo ""
echo "  # With wandb cloud logging:"
echo "  WANDB_MODE=online WANDB_API_KEY=xxx python3 scripts/train.py --num-envs 32"
echo ""
