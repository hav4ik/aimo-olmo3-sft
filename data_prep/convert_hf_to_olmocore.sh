#!/bin/bash
# Convert a HuggingFace Olmo 3 model into an OLMo-core distributed checkpoint that
# the SFT trainer can load. OLMo-core does NOT load HF models directly — this step
# is mandatory before training. Weights only (optimizer state can't be recovered).
#
# The model id is resolved from the HF Hub (cached under /data/hf_cache). Validation
# is skipped by default: it would load BOTH the HF and OLMo-core 7B models at once,
# which is too much for a 62 GB-RAM box. We validate end-to-end via training-loss
# parity against the axolotl run instead.
set -euo pipefail

HOST_RUNS_DIR="${HOST_RUNS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"
IMAGE="${IMAGE:-olmo-core-sft:cu130}"   # the one real image (AI2 base + our source)

HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_ARCH="${MODEL_ARCH:-olmo3_7b}"
OUT="${OUT:-/data/checkpoints/olmocore-olmo3-7b-think}"

docker run --rm \
    --env-file "$HOST_RUNS_DIR/.env" \
    -v "$HOST_RUNS_DIR":/runs -v "$HOST_DATA_MOUNT":/data \
    --shm-size=8g \
    "$IMAGE" \
    python /workspace/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py \
        --checkpoint-input-path "$HF_MODEL" \
        --model-arch "$MODEL_ARCH" \
        --tokenizer dolma2 \
        --output-dir "$OUT" \
        --skip-validation
