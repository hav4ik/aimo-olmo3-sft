#!/bin/bash
# In-container OLMo-core SFT run (we're already inside the container — no `docker run`).
# Attention backend is auto-picked from the GPU arch: sm_90 (H100/H200) -> flash_3 (FA3);
# anything else incl. sm_120 (RTX PRO 6000 / Blackwell) -> flash_2 (we built FA2 with
# sm_120). Override any of the env below. Data + checkpoints under /data/training.
#
#   STAGE=convert  -> convert the HF Olmo-3-7B-Think into an OLMo-core distcp checkpoint
#                     (run ONCE before training; output under /data/training/checkpoints)
#   STAGE=train    -> (default) run SFT. PRECISION=bf16|fp8. fp8 => rowwise + all-attn BF16.
# A single-GPU smoke needs small shapes: SEQ_LEN=2048 GLOBAL_BATCH_SIZE=4096 EPOCHS unset
#   MAX_STEPS=10 (full recipe defaults are seq 32768 / 1,048,576 tok / 2 epochs, multi-GPU).
set -euo pipefail
DATA=/data/training
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}"
CKPT="${CKPT:-$DATA/checkpoints/olmocore-olmo3-7b-think/model_and_optim}"
CKPT_DIR="$(dirname "$CKPT")"

if [ "${STAGE:-train}" = "convert" ]; then
    echo "[olmocore] converting $HF_MODEL -> $CKPT_DIR"
    exec python /workspace/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py \
        --checkpoint-input-path "$HF_MODEL" --model-arch "${MODEL_ARCH:-olmo3_7b}" \
        --tokenizer dolma2 --output-dir "$CKPT_DIR" --skip-validation
fi

PRECISION="${PRECISION:-bf16}"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$CC" in 9.*) DEFATTN=flash_3 ;; *) DEFATTN=flash_2 ;; esac
export OLMO_ATTN_BACKEND="${OLMO_ATTN_BACKEND:-$DEFATTN}"
export OLMO_SFT_SAVE_ROOT="$DATA/checkpoints"
[ "$PRECISION" = "fp8" ] && export OLMO_FP8="${OLMO_FP8:-rowwise}"   # rowwise + all-attn BF16 (DeepSeek)

# prepped by data_prep/prepare.sh --name $DATASET_NAME (-> $DATA/datasets/$NAME/olmocore)
DATASET="${DATASET:-$DATA/datasets/${DATASET_NAME:-tulu-math}/olmocore}"
RUN_NAME="${RUN_NAME:-olmo3-7b-sft-$PRECISION}"
DUR_UNIT="${DUR_UNIT:-epochs}"; DUR_VAL="${EPOCHS:-2}"
[ -n "${MAX_STEPS:-}" ] && { DUR_UNIT=steps; DUR_VAL="$MAX_STEPS"; }

# Single node by default; multi-node (e.g. 16xH200) via NNODES + NODE_RANK + HEAD_NODE_IP.
NNODES="${NNODES:-1}"
if [ "$NNODES" -gt 1 ]; then
    RDZV=(--nnodes="$NNODES" --node_rank="${NODE_RANK:-0}" --rdzv_id="$RUN_NAME" \
          --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE_IP:-127.0.0.1}:${NCCL_PORT:-29400}")
else
    RDZV=(--standalone --nnodes=1)
fi

echo "[olmocore] $PRECISION | ${NNODES}x${NPROC} GPU cc=$CC | attn=$OLMO_ATTN_BACKEND | fp8=${OLMO_FP8:-off} | $DUR_VAL $DUR_UNIT | data=$DATASET"
exec torchrun "${RDZV[@]}" --nproc_per_node="$NPROC" \
    /workspace/code/olmocore/sft_scripts/Olmo-3-7B-SFT-local.py \
    train "$RUN_NAME" "$CKPT" "${CLUSTER:-local_h100}" \
    --seq_len="${SEQ_LEN:-32768}" --num_nodes="$NNODES" \
    --global_batch_size="${GLOBAL_BATCH_SIZE:-1048576}" \
    --dataset_path="$DATASET" \
    --train_module.optim.lr="${LR:-5e-5}" \
    --trainer.max_duration.value="$DUR_VAL" --trainer.max_duration.unit="$DUR_UNIT"
