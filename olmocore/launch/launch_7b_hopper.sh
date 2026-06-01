#!/bin/bash
# Olmo 3 7B SFT on Hopper (H100/H200) — REAL training, via AI2's official OLMo-core
# image + their actual SFT script (Olmo-3-7B-SFT-local.py, a beaker-stubbed copy of
# Olmo-3-7B-SFT.py). We pass the SAME CLI overrides AI2's open-instruct
# `scripts/train/olmo3/7b_think_sft.sh` uses, so this is their recipe verbatim:
#   lr 5e-5 (think), 2 epochs, global_batch 1,048,576 tokens, seq 32768,
#   SkipStepAdamW, hsdp, selective AC, compile, YaRN.
#
# H200 PERFORMANCE PROFILE (layered on AI2's correctness baseline; all overridable):
#   OLMO_ATTN_BACKEND=flash_3  - FA3 packed-varlen attention (faster than AI2's flash_2)
#   OLMO_FP8=""  (DEFAULT: off / bf16)  - matches BOTH reference recipes: AI2 trains
#                                Olmo 3 SFT in bf16, and NVIDIA post-trains Nemotron Nano in
#                                bf16 then PTQs to FP8 only for INFERENCE. FP8 *training* is a
#                                pretraining/throughput tool (with attention + first/last
#                                layers kept high-precision), risky for a short SFT. To
#                                experiment: OLMO_FP8=tensorwise (fastest) | rowwise (native
#                                CUTLASS, more accurate) — exclude lm_head/embeddings, and
#                                VALIDATE loss parity vs bf16 first. FP8's real home is serving.
#   context parallelism        - AUTOMATIC: at seq 32768 the script sets cp_degree=2
#                                (>16384 tok/rank), llama3 ring attention with doc masking.
# NOTE: FP8 changes numerics — validate loss parity vs bf16 on a 2xH100 short run first.
# For an exact-AI2 (bf16/flash_2) baseline run, set OLMO_ATTN_BACKEND="" OLMO_FP8="".
#
# ONE launcher, four shapes (set via env). world_size must be a power of 2.
#   1x H100 :  NPROC_PER_NODE=1  NNODES=1                                  # smoke only; 7B AdamW (~84GB) is tight on one 80GB card
#   2x H100 :  NPROC_PER_NODE=2  NNODES=1                                  # ~42 GB/GPU
#   8x H200 :  NPROC_PER_NODE=8  NNODES=1
#   16xH200 :  NPROC_PER_NODE=8  NNODES=2  NODE_RANK=0|1  HEAD_NODE_IP=...
set -euo pipefail

HOST_RUNS_DIR="${HOST_RUNS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HOST_DATA_MOUNT="${HOST_DATA_MOUNT:-/mnt/data}"
IMAGE="${IMAGE:-olmo-core-sft:cu130}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
NCCL_PORT="${NCCL_PORT:-29400}"
CLUSTER="${CLUSTER:-local_h100}"   # gpu_type only; H100/H200 both map to h100

RUN_NAME="${RUN_NAME:-olmo3-7b-sft}"
CKPT="${CKPT:-/data/checkpoints/olmocore-olmo3-7b-think/model_and_optim}"
DATASET="${DATASET:-/data/datasets/tulu-math-olmocore-oi}"

# === AI2's 7b_think_sft recipe (override via env per experiment) ===
LR="${LR:-5e-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1048576}"
EPOCHS="${EPOCHS:-2}"
SEQ_LEN="${SEQ_LEN:-32768}"

if [ "$NNODES" -gt 1 ]; then
    RDZV=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --rdzv_id="$RUN_NAME" \
          --rdzv_backend=c10d --rdzv_endpoint="$HEAD_NODE_IP:$NCCL_PORT")
else
    RDZV=(--standalone --nnodes=1)
fi

docker run --rm --gpus all --shm-size=64g --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e OLMO_SFT_SAVE_ROOT=/data/checkpoints \
    -e OLMO_ATTN_BACKEND="${OLMO_ATTN_BACKEND-flash_3}" \
    -e OLMO_FP8="${OLMO_FP8-}" \
    --env-file "$HOST_RUNS_DIR/.env" \
    -v "$HOST_RUNS_DIR":/runs -v "$HOST_DATA_MOUNT":/data \
    "$IMAGE" \
    torchrun "${RDZV[@]}" --nproc_per_node="$NPROC_PER_NODE" \
        /runs/sft_scripts/Olmo-3-7B-SFT-local.py \
        train "$RUN_NAME" "$CKPT" "$CLUSTER" \
        --seq_len="$SEQ_LEN" --num_nodes="$NNODES" --global_batch_size="$GLOBAL_BATCH_SIZE" \
        --dataset_path="$DATASET" \
        --train_module.optim.lr="$LR" \
        --trainer.max_duration.value="$EPOCHS" --trainer.max_duration.unit=epochs
