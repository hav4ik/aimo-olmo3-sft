#!/bin/bash
# Offline data prep for BOTH frameworks from ONE input. Run on a box with docker + the
# data-prep image (NOT inside a Vast.AI training container). Produces, under
# <DATA>/training/datasets/<NAME>/ :
#     messages.parquet   - normalized, shared (Axolotl reads this directly)
#     olmocore/          - tokenized .npy for OLMo-core (open-instruct + the chat template)
# Then mount <DATA>/training into the training containers and select with DATASET_NAME=<NAME>.
#
# Usage:
#   ./prepare.sh --name <NAME> --input <SPEC> [--input <SPEC> ...] \
#       [--data-files "sft/*.parquet"]    # parquet glob inside the HF repo named by --input
#       [--template olmo_thinker]         # OLMo-core chat template (think data: olmo_thinker;
#                                         #   non-think data on a thinker model: olmo_thinker_no_think_sft_tokenization)
#       [--max-seq 32768] [--max-examples N] \
#       [--prompt-field q --response-field a --system-field s]   # if no `messages` column
#
# <SPEC> is any of: HF id (org/name) | local parquet glob/list | hf://datasets/org/name/*.parquet
set -euo pipefail
HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"          # host dir mounted to /data
DATAPREP_IMG="${DATAPREP_IMG:-open-instruct-dataprep:0.1.0}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Think}"
HERE="$(cd "$(dirname "$0")" && pwd)"

NAME=""; TEMPLATE="olmo_thinker"; MAXSEQ="32768"; DATA_FILES=""
INPUT_ARGS=(); MAP_ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --name)           NAME="$2"; shift 2;;
  --input)          INPUT_ARGS+=(--input "$2"); shift 2;;
  --data-files)     DATA_FILES="$2"; shift 2;;
  --template)       TEMPLATE="$2"; shift 2;;
  --max-seq)        MAXSEQ="$2"; shift 2;;
  --max-examples)   MAP_ARGS+=(--max-examples "$2"); shift 2;;
  --prompt-field)   MAP_ARGS+=(--prompt-field "$2"); shift 2;;
  --response-field) MAP_ARGS+=(--response-field "$2"); shift 2;;
  --system-field)   MAP_ARGS+=(--system-field "$2"); shift 2;;
  --messages-field) MAP_ARGS+=(--messages-field "$2"); shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done
[ -n "$NAME" ] || { echo "ERROR: --name required"; exit 2; }
[ ${#INPUT_ARGS[@]} -gt 0 ] || { echo "ERROR: at least one --input required"; exit 2; }
[ -n "$DATA_FILES" ] && INPUT_ARGS+=(--data-files "$DATA_FILES")

OUT="/data/training/datasets/$NAME"
DOCKER=(docker run --rm -e HF_TOKEN="${HF_TOKEN:-}" -v "$HOST_DATA_MOUNT":/data --shm-size=8g)

echo "==> [1/2] normalize -> $OUT/messages.parquet"
"${DOCKER[@]}" -v "$HERE":/prep "$DATAPREP_IMG" \
    python /prep/normalize.py "${INPUT_ARGS[@]}" "${MAP_ARGS[@]}" --output "$OUT/messages.parquet"

echo "==> [2/2] OLMo-core tokenize -> $OUT/olmocore   (template=$TEMPLATE, max_seq=$MAXSEQ)"
"${DOCKER[@]}" "$DATAPREP_IMG" \
    python /workspace/open-instruct/scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list "$OUT/messages.parquet" 1.0 \
        --tokenizer_name_or_path "$TOKENIZER" \
        --chat_template_name "$TEMPLATE" \
        --output_dir "$OUT/olmocore" \
        --dataset_local_cache_dir /data/oi_cache \
        --max_seq_length "$MAXSEQ" --num_examples 0 --visualize True

echo ""
echo "==> done. Train with DATASET_NAME=$NAME :"
echo "    OLMo-core reads  $OUT/olmocore   |   Axolotl reads  $OUT/messages.parquet"
