#!/bin/bash
# Tokenize SFT data into OLMo-core .npy using OLMo's OWN canonical tool:
# open-instruct's convert_sft_data_for_olmocore.py with the registered `olmo`
# chat template. We deliberately do NOT hand-roll tokenization/masking — Olmo 3
# was post-trained with this exact tool, so it is the source of truth (correct
# masking: assistant content + eos trained, system/user/assistant-header masked;
# documents delimited by the <|endoftext|> eos for packing).
#
# Runs in the dedicated open-instruct DATA-PREP image (see
# ../../open-instruct/Dockerfile.dataprep), which installs open-instruct's
# package + just the conversion deps (no vllm/flash-attn).
set -euo pipefail

HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"
HOST_RUNS_DIR="${HOST_RUNS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${IMAGE:-open-instruct-dataprep:0.1.0}"

DATASET="${DATASET:-allenai/tulu-3-sft-personas-math}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Think}"   # dolma2 vocab, matches the model
OUT="${OUT:-/data/datasets/tulu-math-olmocore-oi}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"   # smoke; use 32768 for real runs (Olmo recommends 32k)
NUM_EXAMPLES="${NUM_EXAMPLES:-2000}" # 0 = all

docker run --rm \
    --env-file "$HOST_RUNS_DIR/.env" \
    -v "$HOST_DATA_MOUNT":/data \
    --shm-size=8g \
    "$IMAGE" \
    python /workspace/open-instruct/scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list "$DATASET" 1.0 \
        --tokenizer_name_or_path "$TOKENIZER" \
        --chat_template_name olmo \
        --output_dir "$OUT" \
        --dataset_local_cache_dir /data/oi_cache \
        --max_seq_length "$MAX_SEQ_LEN" \
        --num_examples "$NUM_EXAMPLES" \
        --visualize True
