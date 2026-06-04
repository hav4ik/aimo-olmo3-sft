#!/bin/bash
# In-container Axolotl SFT run. Loads a PRE-TOKENIZED parquet (input_ids/labels/attention_mask) from
# HF (DATASET_HF, set by EXPERIMENT) — the SAME dolma2 tokens as the olmo-core run, so the two engines
# are apples-to-apples. axolotl auto-detects already-tokenized data and SKIPS online tokenization (only
# the multipack packing prep runs, cached under /data/training/last_run_prepared). The base model
# downloads from HF (HF_TOKEN). attn auto by GPU arch: sm_90 (H100/H200) -> flash_attention_3; else
# (sm_120 RTX PRO 6000 / Blackwell) -> flex_attention. Ckpts under /data/training.
#   MODEL_SIZE=7b|32b + PRECISION=bf16|fp8 -> configs/olmo3-<size>-<precision>.yaml ; DATASET_NAME = data.
#   Single-GPU smoke: SEQUENCE_LEN=2048 MAX_STEPS=10 (full recipe = seq 65536 / 2 epochs).
set -euo pipefail
[ -f /workspace/axolotl-venv/bin/activate ] && source /workspace/axolotl-venv/bin/activate
DATA=/data/training
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
PRECISION="${PRECISION:-bf16}"
MODEL_SIZE="${MODEL_SIZE:-7b}"   # 7b | 32b (32b = the axolotl fallback path for the big tier)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # works whether code is cloned or baked
CONFIG="${CONFIG:-$HERE/configs/olmo3-${MODEL_SIZE}-${PRECISION}.yaml}"
# Dataset = a PRE-TOKENIZED parquet. DATASET_SRC defaults to the HF dataset repo (DATASET_HF, set by
# EXPERIMENT) and DATASET_FILE to the parquet inside it (axolotl/sft_tokenized.parquet). HF load_dataset
# infers parquet from the .parquet extension and pulls ONLY that file (not the olmocore .npy). Override
# DATASET_SRC with a local .parquet (data_files=null) or a local dir holding DATASET_FILE to use a
# pre-staged copy; point DATASET_SRC at any other HF id for a smoke set.
DATASET_SRC="${DATASET_SRC:-${DATASET_HF:-}}"
[ -n "$DATASET_SRC" ] || { echo "ERROR: no dataset — set EXPERIMENT, or DATASET_HF / DATASET_SRC=<hf-dataset-id|local.parquet>"; exit 3; }
DATASET_FILE="${DATASET_FILE:-axolotl/sft_tokenized.parquet}"   # parquet within the HF repo / local dir
case "$DATASET_SRC" in
    *.parquet) DATASET_PATH="$DATASET_SRC"; DATA_FILES="null" ;;          # explicit parquet file -> path IS the file
    *)         DATASET_PATH="$DATASET_SRC"; DATA_FILES="$DATASET_FILE" ;; # HF repo (or local dir) + the parquet inside it
esac

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$CC" in 9.*) DEFATTN=flash_attention_3 ;; *) DEFATTN=flex_attention ;; esac
ATTN="${ATTN_IMPL:-$DEFATTN}"
# Sequence parallelism for long context (e.g. 65536): CONTEXT_PARALLEL_SIZE>1 (a divisor of the total
# GPUs) splits each sequence across that many ranks so the activations fit. SP needs flash attention
# (its ring kernel is FA2-based; FA3 is Hopper-only), so force flash_attention_2 when on. ring-flash-attn
# also requires micro_batch_size=1 + sample_packing, which the configs already set.
CP_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
[ "$CP_SIZE" -gt 1 ] && ATTN=flash_attention_2

# materialize the config with the dataset path + data_files substituted in
CFG=/tmp/axolotl-config.yaml
sed -e "s|__DATASET__|$DATASET_PATH|g" -e "s|__DATA_FILES__|$DATA_FILES|g" "$CONFIG" > "$CFG"

# torch.compile toggle. FP8 wants compile for the scaled_mm fusion speedup, BUT compiling HF's RoPE
# (an inv_freq @ position_ids outer product, k=1 + stride-0 broadcast) makes inductor emit a cuBLAS
# Sgemm with ldb=0 at long context -> CUBLAS_STATUS_INVALID_VALUE. TORCH_COMPILE=false runs eager and
# dodges it (bf16: ~free; fp8: still trains via torchao Float8Linear, just unfused/slower). Keep compile
# but route gemms through Triton instead of cuBLAS to ALSO dodge it:
#   -e TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 -e TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON  (slow compile)
if [ -n "${TORCH_COMPILE:-}" ]; then
    sed -i "s|^torch_compile:.*|torch_compile: ${TORCH_COMPILE}|" "$CFG"
    echo "[axolotl] torch_compile override -> ${TORCH_COMPILE}"
fi

OVERRIDES=(
    "--attn_implementation=$ATTN"
    "--output_dir=$DATA/checkpoints/olmo3-${MODEL_SIZE}-axolotl-$PRECISION"
    "--dataset_prepared_path=$DATA/last_run_prepared"
)
[ -n "${SEQUENCE_LEN:-}" ] && OVERRIDES+=("--sequence_len=$SEQUENCE_LEN")
[ -n "${MAX_STEPS:-}"    ] && OVERRIDES+=("--max_steps=$MAX_STEPS")
[ "$CP_SIZE" -gt 1 ]       && OVERRIDES+=("--context_parallel_size=$CP_SIZE")
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
echo "[axolotl] $PRECISION | ${NNODES}x${NPROC} GPU cc=$CC | attn=$ATTN | node ${NODE_RANK}/${NNODES} | config=$(basename "$CONFIG") | data=$DATASET_PATH${DATA_FILES:+ ($DATA_FILES)} [pretokenized]"
exec accelerate launch "${LAUNCH[@]}" -m axolotl.cli.train "$CFG" "${OVERRIDES[@]}"
