#!/bin/bash
# Convert a HuggingFace Olmo 3 model into an OLMo-core distributed (distcp) checkpoint that the
# SFT trainer can load. OLMo-core does NOT load HF models directly — mandatory before training.
# Weights only. (On Vast.AI prefer `STAGE=convert` on the deploy image — same thing, in-container.)
# Validation skipped by default (it loads BOTH HF + OLMo-core models at once; too much RAM).
set -euo pipefail

HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"
IMAGE="${IMAGE:-olmo-core-sft:cu130}"
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_ARCH="${MODEL_ARCH:-olmo3_7b}"
# Default OUT MUST match olmocore/run.sh's CKPT dir (/data/training/checkpoints/...).
OUT="${OUT:-/data/training/checkpoints/olmocore-olmo3-7b-think}"

docker run --rm --shm-size=8g \
    -e HF_TOKEN="${HF_TOKEN:-}" -e HF_MODEL="$HF_MODEL" -e MODEL_ARCH="$MODEL_ARCH" -e OUT="$OUT" \
    -v "$HOST_DATA_MOUNT":/data \
    "$IMAGE" \
    bash -lc '
      # transformer_engine import needs libnvrtc on the ld path (cu13 wheel ships it off-path);
      # without this, convert_checkpoint_from_hf.py crashes at `import olmo_core` (same as run.sh).
      ldconfig -p 2>/dev/null | grep -q "libnvrtc\.so" || { d="$(dirname "$(find /opt/conda -name "libnvrtc.so*" 2>/dev/null | head -1)")"; [ -n "$d" ] && echo "$d" >/etc/ld.so.conf.d/zz-nvrtc.conf && ldconfig; }
      python /workspace/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py \
        --checkpoint-input-path "$HF_MODEL" --model-arch "$MODEL_ARCH" \
        --tokenizer dolma2 --output-dir "$OUT" --skip-validation
    '
