#!/bin/bash
# Launch an Olmo 3 axolotl SFT run in the axolotl image.
#   Usage: ./launch.sh <config-basename> [num_gpus]
#   ./launch.sh olmo3-7b-bf16 8      # BF16 baseline arm
#   ./launch.sh olmo3-7b-fp8  8      # FP8 arm (same config except the fp8 block)
#
# Mirrors the OLMo-core launchers so the BF16-vs-FP8 stability A/B is symmetric.
# secrets (HF_TOKEN / WANDB_API_KEY) come from runs/.env via --env-file (never baked).
set -euo pipefail

CONFIG="${1:?usage: launch.sh <config-basename> [num_gpus]}"
NUM_GPUS="${2:-8}"
HOST_RUNS_DIR="${HOST_RUNS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"
IMAGE="${IMAGE:-axolotl-olmo3-sft:0.1.0}"

docker run --rm --gpus all --shm-size=64g --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env-file "$HOST_RUNS_DIR/.env" \
    -v "$HOST_RUNS_DIR":/runs -v "$HOST_DATA_MOUNT":/data \
    "$IMAGE" \
    bash -lc "accelerate launch --num_processes=${NUM_GPUS} -m axolotl.cli.train /runs/configs/${CONFIG}.yaml"
