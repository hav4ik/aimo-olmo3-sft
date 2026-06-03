#!/bin/bash
# In-container OLMo-core SFT run (we're already inside the container — no `docker run`).
# Attention backend is auto-picked from the GPU arch: sm_90 (H100/H200) -> flash_3 (FA3);
# anything else incl. sm_120 (RTX PRO 6000 / Blackwell) -> flash_2 (we built FA2 with
# sm_120). Override any of the env below. Data + checkpoints under /data/training.
#
#   STAGE=train    -> (default) run SFT. PRECISION=bf16|fp8. fp8 => rowwise + all-attn BF16.
#                     Auto-converts the HF base model -> OLMo-core distcp checkpoint on first
#                     run (one-time per /data/training volume, CPU; NODE_RANK=0 converts and
#                     the other nodes wait). So a single `docker run` does convert+train.
#   STAGE=convert  -> just (re)produce the distcp checkpoint and exit (explicit pre-stage).
# Data: olmo-core .npy pulled from HF at runtime (DATASET_HF + DATASET_SUBDIR, set by EXPERIMENT;
#   rank-0 stages on the shared mount). Or pre-stage the .npy under $DATA/datasets/<NAME>/olmocore.
# A single-GPU smoke needs small shapes: SEQ_LEN=2048 GLOBAL_BATCH_SIZE=4096 MAX_STEPS=10
#   (full recipe defaults are seq 65536 / 1,048,576 tok / 2 epochs, multi-GPU).
set -euo pipefail

# transformer_engine's _load_nvrtc() does `ldconfig -p | grep libnvrtc.so` at import;
# the cu13 wheel ships libnvrtc under site-packages/nvidia/cu13/lib which is NOT on the
# default linker path, so the grep returns non-zero and `import olmo_core.nn.attention`
# (pulled in by the SFT script) dies with CalledProcessError before any training starts.
# Put that dir on the ld path so TE imports. Idempotent; needs root (container runs as root).
if ! ldconfig -p 2>/dev/null | grep -q 'libnvrtc\.so'; then
    _nvrtc_dir="$(dirname "$(find /opt/conda -name 'libnvrtc.so*' 2>/dev/null | head -1)")"
    if [ -n "$_nvrtc_dir" ] && [ -w /etc/ld.so.conf.d ]; then
        echo "$_nvrtc_dir" > /etc/ld.so.conf.d/zz-nvrtc.conf && ldconfig
        echo "[olmocore] added $_nvrtc_dir to ld path (libnvrtc/TE fix)"
    fi
fi

DATA=/data/training
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
# Model + recipe by size (the EXPERIMENT/MODEL_SIZE front-end sets MODEL_SIZE; explicit env wins).
MODEL_SIZE="${MODEL_SIZE:-7b}"
case "$MODEL_SIZE" in
    7b)  HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}";    MODEL_ARCH="${MODEL_ARCH:-olmo3_7b}";  DEF_LR=5e-5; DEF_GBS=1048576 ;;
    32b) HF_MODEL="${HF_MODEL:-allenai/Olmo-3.1-32B-Think}"; MODEL_ARCH="${MODEL_ARCH:-olmo3_32b}"; DEF_LR=1e-4; DEF_GBS=4194304 ;;
    *)   echo "ERROR: MODEL_SIZE='$MODEL_SIZE' (want 7b|32b)"; exit 2 ;;
esac
CKPT="${CKPT:-$DATA/checkpoints/olmocore-olmo3-${MODEL_SIZE}-think/model_and_optim}"
CKPT_DIR="$(dirname "$CKPT")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's dir (cloned or baked)
# Staging node for one-time rank-0 work (convert + data prep) on the shared storage. node 0 has
# rank 0 in BOTH conventions (node-level GLOBAL_RANK=0, or process-level base rank 0).
THIS_NODE_RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"

# One-time HF -> OLMo-core distcp conversion, cached on the /data/training volume. OLMo-core
# can't load HF weights directly; the distcp format reshards on LOAD, so this single-process
# convert loads onto any GPU/node count later. Same image converts AND loads => self-consistent
# by construction, so there's no pinned-artifact to drift out of sync with the code. We gate on
# a completion sentinel (not dir-exists) so a crashed/partial convert re-runs cleanly.
# Multi-node: only NODE_RANK=0 converts; the other nodes wait for the sentinel.
CONVERT_DONE="$CKPT_DIR/.convert_complete"
if [ "${STAGE:-train}" = "convert" ] || [ ! -f "$CONVERT_DONE" ]; then
    if [ "$THIS_NODE_RANK" -eq 0 ]; then
        echo "[olmocore] converting $HF_MODEL -> $CKPT_DIR (one-time per volume; CPU)"
        rm -rf "$CKPT" "$CONVERT_DONE"
        python /workspace/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py \
            --checkpoint-input-path "$HF_MODEL" --model-arch "${MODEL_ARCH:-olmo3_7b}" \
            --tokenizer dolma2 --output-dir "$CKPT_DIR" --skip-validation
        touch "$CONVERT_DONE"
        echo "[olmocore] convert complete ($CONVERT_DONE)"
    else
        echo "[olmocore] node_rank=$THIS_NODE_RANK waiting for rank-0 convert ($CONVERT_DONE)…"
        for _ in $(seq 1 720); do [ -f "$CONVERT_DONE" ] && break; sleep 10; done
        [ -f "$CONVERT_DONE" ] || { echo "ERROR: timed out waiting for $CONVERT_DONE"; exit 4; }
    fi
fi
# convert-only mode: checkpoint is ready, don't train.
[ "${STAGE:-train}" = "convert" ] && { echo "[olmocore] STAGE=convert: checkpoint ready, exiting (no training)."; exit 0; }

PRECISION="${PRECISION:-bf16}"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$CC" in 9.*) DEFATTN=flash_3 ;; *) DEFATTN=flash_2 ;; esac
export OLMO_ATTN_BACKEND="${OLMO_ATTN_BACKEND:-$DEFATTN}"
export OLMO_SFT_SAVE_ROOT="$DATA/checkpoints"
[ "$PRECISION" = "fp8" ] && export OLMO_FP8="${OLMO_FP8:-rowwise}"   # rowwise + all-attn BF16 (DeepSeek)

# Optimizer per precision: fused AdamW for the stable BF16 baseline (single fused CUDA kernel =
# faster), SkipStepAdamW for FP8 (spike protection + the trainer's `optim/step skipped` metric,
# which averages to the FP8 step-skip frequency). Override explicitly with OLMO_OPTIM.
if [ -z "${OLMO_OPTIM:-}" ]; then
    [ "$PRECISION" = "fp8" ] && OLMO_OPTIM=skip_step || OLMO_OPTIM=fused_adamw
fi
export OLMO_OPTIM

# ---- Data (runtime) ---------------------------------------------------------------------------
# Use prepped .npy on the volume if present; else DOWNLOAD the pre-tokenized olmo-core .npy from HF.
# DATASET_HF = the HF *dataset* repo; DATASET_SUBDIR = subfolder inside it holding the
# token_ids_part_*.npy / labels_mask_*.npy (e.g. "olmocore"; set by EXPERIMENT). rank-0 stages on
# shared storage; other nodes wait. Pin with DATASET_REVISION.
DL_DIR="$DATA/datasets/${DATASET_NAME:-tulu-math}"
SUB="${DATASET_SUBDIR-olmocore}"
DATASET="${DATASET:-$DL_DIR${SUB:+/$SUB}}"
if ! ls "$DATASET"/token_ids_part_*.npy >/dev/null 2>&1; then
    [ -n "${DATASET_HF:-}" ] || { echo "ERROR: no tokenized data in $DATASET and DATASET_HF unset (set EXPERIMENT, or DATASET_HF=<hf-dataset-repo>)"; exit 3; }
    DATA_READY="$DL_DIR/.data_ready"
    if [ "$THIS_NODE_RANK" -eq 0 ]; then
        echo "[olmocore] downloading $DATASET_HF${DATASET_REVISION:+@$DATASET_REVISION} (subdir: ${SUB:-/}) -> $DL_DIR"
        rm -f "$DATA_READY"
        HF_ARGS=(--repo-type dataset --local-dir "$DL_DIR")
        [ -n "$SUB" ] && HF_ARGS+=(--include "$SUB/*")
        [ -n "${DATASET_REVISION:-}" ] && HF_ARGS+=(--revision "$DATASET_REVISION")
        hf download "$DATASET_HF" "${HF_ARGS[@]}"
        ls "$DATASET"/token_ids_part_*.npy >/dev/null 2>&1 || { echo "ERROR: $DATASET_HF (subdir ${SUB:-/}) has no token_ids_part_*.npy at $DATASET"; exit 3; }
        touch "$DATA_READY"
        echo "[olmocore] data ready: $DATASET"
    else
        echo "[olmocore] node_rank=$THIS_NODE_RANK waiting for rank-0 data download ($DATA_READY)…"
        for _ in $(seq 1 720); do [ -f "$DATA_READY" ] && break; sleep 10; done
        [ -f "$DATA_READY" ] || { echo "ERROR: timed out waiting for $DATA_READY"; exit 3; }
    fi
fi
RUN_NAME="${RUN_NAME:-olmo3-${MODEL_SIZE}-sft-$PRECISION}"
DUR_UNIT="${DUR_UNIT:-epochs}"; DUR_VAL="${EPOCHS:-2}"
[ -n "${MAX_STEPS:-}" ] && { DUR_UNIT=steps; DUR_VAL="$MAX_STEPS"; }

# Optional overrides appended to the train command (off by default => AI2 auto-derivation).
#  RANK_MICROBATCH_TOKENS: per-DP-rank microbatch in TOKENS, must be a multiple of SEQ_LEN
#    (262144 = 4x65536 = 4 sequences/microstep). By default BatchSizeConfig caps this at
#    16384*cp (= 1 sequence) and pads out the rest with grad-accum. Raise it to pack more
#    sequences per microstep and spend spare VRAM on throughput. We do NOT set grad-accum:
#    the trainer recomputes it from global_batch_size, so GBS stays fixed at 1M across any
#    node count / microbatch (grad_accum = GBS / (this * dp_world_size); must be a +integer).
#    With cp=4 the per-GPU activation is this/cp tokens (262144 -> 65536 tok/GPU, 4x default).
EXTRA=()
[ -n "${RANK_MICROBATCH_TOKENS:-}" ] && EXTRA+=("--train_module.rank_microbatch_size=$RANK_MICROBATCH_TOKENS")

# ---- Multi-node launch ------------------------------------------------------------------------
# The container owns the whole node and torchrun spawns one rank per local GPU (--nproc_per_node);
# torchrun sets each rank's RANK/LOCAL_RANK/WORLD_SIZE. Node topology, in priority order:
#   1. ABCI/PBS env (MASTER_ADDR + MASTER_PORT + WORLD_SIZE + GLOBAL_RANK) -> static torchrun.
#      ASSUMPTION: WORLD_SIZE = number of NODES, GLOBAL_RANK = this node's rank (0-indexed) — the
#      "one container per node, local rank handled inside" model. If your scheduler sets these as
#      process-level counts instead, pass NNODES/NODE_RANK explicitly (they take precedence).
#   2. NNODES>1 + HEAD_NODE_IP -> c10d rendezvous (our Vast.AI multi-node path).
#   3. single node -> standalone.
# Resolve node topology from the scheduler env. Handles BOTH conventions, auto-detected by whether
# WORLD_SIZE is a clean multiple of the local GPU count ($NPROC):
#   - process-level (standard torchrun): WORLD_SIZE = total ranks, GLOBAL_RANK = this node's base rank
#   - node-level:                        WORLD_SIZE = #nodes,      GLOBAL_RANK = node index
# Override either with NNODES / NODE_RANK.
if [ -n "${NNODES:-}" ]; then
    NODE_RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"
elif [ -n "${WORLD_SIZE:-}" ] && [ "$WORLD_SIZE" -gt "$NPROC" ] && [ $((WORLD_SIZE % NPROC)) -eq 0 ]; then
    NNODES=$((WORLD_SIZE / NPROC)); NODE_RANK="${NODE_RANK:-$(( ${GLOBAL_RANK:-0} / NPROC ))}"
else
    NNODES="${WORLD_SIZE:-1}"; NODE_RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"
fi
if [ -n "${MASTER_ADDR:-}" ] && [ "$NNODES" -gt 1 ]; then
    RDZV=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
          --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29400}")
elif [ "$NNODES" -gt 1 ]; then
    RDZV=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --rdzv_id="$RUN_NAME"
          --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE_IP:-127.0.0.1}:${NCCL_PORT:-29400}")
else
    RDZV=(--standalone --nnodes=1)
fi
# torchrun assigns the children's ranks; drop inherited process-level vars so they can't shadow it.
unset RANK WORLD_SIZE GLOBAL_RANK LOCAL_RANK 2>/dev/null || true

echo "[olmocore] $PRECISION | ${NNODES}x${NPROC} GPU cc=$CC | attn=$OLMO_ATTN_BACKEND | fp8=${OLMO_FP8:-off} | optim=$OLMO_OPTIM | node ${NODE_RANK}/${NNODES} | $DUR_VAL $DUR_UNIT"
# Size-specific SFT script (local copy of AI2's, beaker-stubbed): Olmo-3-7B/32B-SFT-local.py.
SFT_SCRIPT="$HERE/sft_scripts/Olmo-3-${MODEL_SIZE^^}-SFT-local.py"
[ -f "$SFT_SCRIPT" ] || { echo "ERROR: no SFT script for MODEL_SIZE=$MODEL_SIZE at $SFT_SCRIPT"; exit 2; }
exec torchrun "${RDZV[@]}" --nproc_per_node="$NPROC" \
    "$SFT_SCRIPT" \
    train "$RUN_NAME" "$CKPT" "${CLUSTER:-local_h100}" \
    --seq_len="${SEQ_LEN:-65536}" --num_nodes="$NNODES" \
    --global_batch_size="${GLOBAL_BATCH_SIZE:-$DEF_GBS}" \
    --dataset_path="$DATASET" \
    --train_module.optim.lr="${LR:-$DEF_LR}" \
    --trainer.max_duration.value="$DUR_VAL" --trainer.max_duration.unit="$DUR_UNIT" \
    ${EXTRA[@]+"${EXTRA[@]}"}
