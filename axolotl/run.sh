#!/bin/bash
# In-container Axolotl SFT run. Reads the OFFLINE-prepped local parquet produced by
# data_prep/prepare.sh --name $DATASET_NAME  (-> /data/training/datasets/$NAME/messages.parquet);
# the base model downloads from HF (HF_TOKEN). attn auto by GPU arch: sm_90 (H100/H200) ->
# flash_attention_3; else (sm_120 RTX PRO 6000 / Blackwell) -> flex_attention. Ckpts under /data/training.
#   MODEL_SIZE=7b|32b + PRECISION=bf16|fp8 -> configs/olmo3-<size>-<precision>.yaml ; DATASET_NAME = data.
#   Single-GPU smoke: SEQUENCE_LEN=2048 MAX_STEPS=10 (full recipe = seq 32768 / 2 epochs).
set -euo pipefail
[ -f /workspace/axolotl-venv/bin/activate ] && source /workspace/axolotl-venv/bin/activate
DATA=/data/training
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
PRECISION="${PRECISION:-bf16}"
MODEL_SIZE="${MODEL_SIZE:-7b}"   # 7b | 32b (32b = the axolotl fallback path for the big tier)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # works whether code is cloned or baked
CONFIG="${CONFIG:-$HERE/configs/olmo3-${MODEL_SIZE}-${PRECISION}.yaml}"
PARQUET="${DATASET_PARQUET:-$DATA/datasets/${DATASET_NAME:-tulu-math}/messages.parquet}"
[ -s "$PARQUET" ] || { echo "ERROR: prepped data missing/empty: $PARQUET — run data_prep/prepare.sh --name ${DATASET_NAME:-tulu-math}"; exit 3; }

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$CC" in 9.*) DEFATTN=flash_attention_3 ;; *) DEFATTN=flex_attention ;; esac
ATTN="${ATTN_IMPL:-$DEFATTN}"

# materialize the config with the prepped parquet path substituted in
CFG=/tmp/axolotl-config.yaml
sed "s|__DATASET_PARQUET__|$PARQUET|g" "$CONFIG" > "$CFG"

OVERRIDES=(
    "--attn_implementation=$ATTN"
    "--output_dir=$DATA/checkpoints/olmo3-${MODEL_SIZE}-axolotl-$PRECISION"
    "--dataset_prepared_path=$DATA/last_run_prepared"
)
[ -n "${SEQUENCE_LEN:-}" ] && OVERRIDES+=("--sequence_len=$SEQUENCE_LEN")
[ -n "${MAX_STEPS:-}"    ] && OVERRIDES+=("--max_steps=$MAX_STEPS")
# FP8 on Blackwell sm_120: flex_attention + torchao-fp8 compile path is untested — for the
# first RTX 6000 run use BF16; FP8 is the proven path on Hopper.

# Multi-node (accelerate): same ABCI/PBS env as olmocore — one container per node, NPROC ranks each.
# WORLD_SIZE = #nodes, GLOBAL_RANK = node rank; override with NNODES/NODE_RANK. Single node: NPROC.
# Same topology resolution as olmocore (auto-handles process-level vs node-level WORLD_SIZE).
if [ -n "${NNODES:-}" ]; then
    NODE_RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"
elif [ -n "${WORLD_SIZE:-}" ] && [ "$WORLD_SIZE" -gt "$NPROC" ] && [ $((WORLD_SIZE % NPROC)) -eq 0 ]; then
    NNODES=$((WORLD_SIZE / NPROC)); NODE_RANK="${NODE_RANK:-$(( ${GLOBAL_RANK:-0} / NPROC ))}"
else
    NNODES="${WORLD_SIZE:-1}"; NODE_RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"
fi
if [ "$NNODES" -gt 1 ]; then
    LAUNCH=(--num_machines="$NNODES" --machine_rank="$NODE_RANK"
            --main_process_ip="${MASTER_ADDR:-${HEAD_NODE_IP:-127.0.0.1}}"
            --main_process_port="${MASTER_PORT:-29400}"
            --num_processes="$((NPROC * NNODES))")
else
    LAUNCH=(--num_processes="$NPROC")
fi
unset RANK WORLD_SIZE GLOBAL_RANK LOCAL_RANK 2>/dev/null || true
echo "[axolotl] $PRECISION | ${NNODES}x${NPROC} GPU cc=$CC | attn=$ATTN | node ${NODE_RANK}/${NNODES} | config=$(basename "$CONFIG")"
exec accelerate launch "${LAUNCH[@]}" -m axolotl.cli.train "$CFG" "${OVERRIDES[@]}"
