#!/bin/bash
# Deploy Catan AI training to RunPod
#
# Prerequisites:
#   - Docker installed
#   - RunPod account with API key
#   - runpodctl CLI installed (pip install runpod)
#
# Usage:
#   ./scripts/deploy_runpod.sh [build|push|create-pod|create-serverless]

set -euo pipefail

IMAGE_NAME="catan-ai"
IMAGE_TAG="latest"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
FULL_IMAGE="${REGISTRY}/${DOCKER_USERNAME:-}/${IMAGE_NAME}:${IMAGE_TAG}"

case "${1:-help}" in
  build)
    echo "Building Docker image..."
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    echo "Done. Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    ;;

  push)
    echo "Tagging and pushing to registry..."
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    echo "Pushed: ${FULL_IMAGE}"
    ;;

  create-pod)
    echo "Creating RunPod GPU pod for training..."
    echo ""
    echo "Go to https://www.runpod.io/console/pods and create a pod with:"
    echo "  - GPU: RTX 3090 or A100 (24GB+ VRAM)"
    echo "  - Image: ${FULL_IMAGE}"
    echo "  - Disk: 50GB+"
    echo "  - Docker command: python scripts/train.py --total-steps 5000000"
    echo ""
    echo "Or use runpodctl:"
    echo "  runpodctl create pod \\"
    echo "    --name catan-train \\"
    echo "    --gpuType 'NVIDIA RTX 3090' \\"
    echo "    --imageName '${FULL_IMAGE}' \\"
    echo "    --volumeSize 50 \\"
    echo "    --args 'python scripts/train.py --total-steps 5000000 --num-envs 16'"
    ;;

  create-serverless)
    echo "Creating RunPod serverless endpoint..."
    echo ""
    echo "Go to https://www.runpod.io/console/serverless and create an endpoint with:"
    echo "  - Image: ${FULL_IMAGE}"
    echo "  - Handler: runpod_handler.py"
    echo "  - GPU: RTX 3090+"
    echo "  - Max workers: 1 (for training) or more (for inference)"
    echo ""
    echo "Submit training job:"
    echo '  curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \'
    echo '    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"input": {"mode": "train", "config": {"total_timesteps": 5000000}}}'"'"
    ;;

  *)
    echo "Usage: $0 [build|push|create-pod|create-serverless]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image locally"
    echo "  push               Push image to container registry"
    echo "  create-pod         Instructions to create a RunPod GPU pod"
    echo "  create-serverless  Instructions to create a RunPod serverless endpoint"
    ;;
esac
