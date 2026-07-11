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

# Default the CUDA allocator to expandable segments — reclaims fragmentation (the "reserved but
# unallocated" memory that builds up over a long run and causes late OOMs). Safe: allocator-only,
# bit-identical training, no cudagraphs in our stack. A caller-provided value still wins.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

DATA="${DATA:-/data/training}"   # overridable: Fields train.py points this at --workdir (downloads/scratch)
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"

# ---- Distributed contract: the launcher MUST set all four; we never infer topology ------------
# WORLD_SIZE = number of NODES, GLOBAL_RANK = this node's 0-based index, MASTER_ADDR/MASTER_PORT =
# rendezvous. Single node: WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400.
# Fields train.py validates these first (friendlier message); this also guards the direct /
# entrypoint.sh path so a missing var fails LOUD here instead of silently running standalone.
: "${WORLD_SIZE:?[olmocore] not set — launcher must set WORLD_SIZE(#nodes) GLOBAL_RANK(node-index) MASTER_ADDR MASTER_PORT. Single node: WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400. Under singularity --containall forward via --env / APPTAINERENV_*}"
: "${GLOBAL_RANK:?[olmocore] not set — this node's 0-based index in [0, WORLD_SIZE)}"
: "${MASTER_ADDR:?[olmocore] not set — rendezvous host (node 0's address)}"
: "${MASTER_PORT:?[olmocore] not set — rendezvous port}"
# Model + recipe by size (the EXPERIMENT/MODEL_SIZE front-end sets MODEL_SIZE; explicit env wins).
MODEL_SIZE="${MODEL_SIZE:-32b}"   # this branch trains the 32B sink deploy model by default
# DEF_KEEP = default cap on persistent checkpoints kept on disk (distcp ~100 GB/7B, ~251 GB/32B). On a
# ~1 TB disk shared with a ~200 GB dataset, the 32B can only afford 2 checkpoints total: 32B keep 1
# (1 persistent ~251 GB + 1 rotating ephemeral ~251 GB = ~502 GB, leaving room for data + an in-flight
# save). 7B keep 2 (~200 GB, it has headroom). Override with OLMO_KEEP_LAST_CKPTS (0=all).
# Via fields/train.py this is always set from --keep-last (default 1); DEF_KEEP is only the fallback for
# running run.sh directly. Keep the two in sync.
# DEF_SFT = the size-specific SFT script basename under sft_scripts/ (resolved to a path further down).
# The 7B/32B share the olmo3 long-context script; 1b is a separate LOCAL TEST path on the real published
# allenai/OLMo-2-0425-1B-Instruct (Olmo-2 1B; --model-arch olmo2_1b_v2, native 4096 ctx — see the script's
# header). 1b is for exercising the full pipeline (convert->train->convert->upload) on one small GPU, NOT a
# real recipe: small lr / batch / keep, and a short single-GPU smoke (e.g. SEQ_LEN=4096 GLOBAL_BATCH_SIZE
# small MAX_STEPS=10).
case "$MODEL_SIZE" in
    1b)  HF_MODEL="${HF_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"; MODEL_ARCH="${MODEL_ARCH:-olmo2_1b_v2}"; DEF_LR=5e-5; DEF_GBS=4096;    DEF_KEEP=1; DEF_SFT=Olmo-2-1B-SFT-local.py;  DEF_SEQ_LEN=4096 ;;
    7b)  HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think}";    MODEL_ARCH="${MODEL_ARCH:-olmo3_7b}";  DEF_LR=5e-5; DEF_GBS=1572864; DEF_KEEP=2; DEF_SFT=Olmo-3-7B-SFT-local.py;  DEF_SEQ_LEN=65536 ;;  # GBS 1.5M == 32B (divides for WORLD_SIZE 2/3/4/6 at cp=4)
    32b) HF_MODEL="${HF_MODEL:-chankhavu/yccchen-olmo3-deploy}"; MODEL_ARCH="${MODEL_ARCH:-olmo3_32b}"; DEF_LR=5e-5; DEF_GBS=1572864; DEF_KEEP=3; DEF_SFT=Olmo-3-32B-SFT-bf16.py; DEF_SEQ_LEN=65536 ;;  # FP8-free bf16 sink recipe on the deploy model (trained sinks + deepseek tok); base=allenai/Olmo-3.1-32B-Think. GBS 1.5M -> divides WORLD_SIZE 2/3/4/6 (cp=4); lr 5e-5 between linear/sqrt of AI2's 1e-4@4.19M
    *)   echo "ERROR: MODEL_SIZE='$MODEL_SIZE' (want 1b|7b|32b)"; exit 2 ;;
esac
# The 32B deploy model (chankhavu/yccchen-olmo3-deploy) is an Olmo-3 WITH per-head attention sinks and a
# DEEPSEEK tokenizer — so for MODEL_SIZE=32b, default sink ON and the tokenizer to the model's own (reuse
# HF_MODEL => deepseek), and the dataset to its tokenized SFT data. Without these a bare-entrypoint launch
# would drop the trained sinks / use the wrong (dolma2) tokenizer. Explicit env always wins; train.py
# already sets all three.
if [ "$MODEL_SIZE" = "32b" ]; then
    OLMO_USE_SINK="${OLMO_USE_SINK:-1}"
    OLMO_HF_TOKENIZER="${OLMO_HF_TOKENIZER:-1}"   # 1 => reuse HF_MODEL's tokenizer (deepseek), NOT dolma2
    DATASET_HF="${DATASET_HF:-chankhavu/yccchen-stage2-olmocore-256k-v2}"
fi
# FP8-FREE 32B recipe: SFT_SCRIPT_NAME=Olmo-3-32B-SFT-bf16.py MODEL_SIZE=32b — a copy of the 32B
# script with all FP8 code removed (the Olmo-3 authors flagged FP8 SFT as too reckless; it raises if
# OLMO_FP8 is set). bootstrap.sh already downloads it with the repo.
SFT_SCRIPT_NAME="${SFT_SCRIPT_NAME:-$DEF_SFT}"   # per-size SFT script basename; explicit env wins
# Tokenizer: default dolma2. OLMO_HF_TOKENIZER=<hf-id> (or 1 => reuse HF_MODEL) makes BOTH the
# HF->distcp convert and training read vocab_size + eos/pad/bos from that model's config.json
# (TokenizerConfig.from_hf) instead of dolma2 — required for non-dolma2 models (e.g. the
# deepseek-transplant Olmo3, vocab 129280). It sizes the embedding/lm_head and gives the eos id for
# intra-doc masking, so it MUST match the base checkpoint AND how the .npy data was tokenized.
[ "${OLMO_HF_TOKENIZER:-}" = "1" ] && OLMO_HF_TOKENIZER="$HF_MODEL"
TOKENIZER_ARG="${OLMO_HF_TOKENIZER:-dolma2}"
[ -n "${OLMO_HF_TOKENIZER:-}" ] && export OLMO_TOKENIZER_HF="$OLMO_HF_TOKENIZER"
SEQ_LEN="${SEQ_LEN:-$DEF_SEQ_LEN}"   # per-size default (1b=4096 native, 7b/32b=65536); explicit env wins
export OLMO_KEEP_LAST_CKPTS="${OLMO_KEEP_LAST_CKPTS:-$DEF_KEEP}"
CKPT="${CKPT:-$DATA/checkpoints/olmocore-olmo3-${MODEL_SIZE}-think/model_and_optim}"
CKPT_DIR="$(dirname "$CKPT")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's dir (cloned or baked)
# Staging node for one-time rank-0 work (convert + data prep) on the shared storage. GLOBAL_RANK is
# this node's index per the explicit contract above; node 0 (GLOBAL_RANK=0) stages, the rest wait.
THIS_NODE_RANK="$GLOBAL_RANK"

# Retry a (resumable) download with exponential backoff. huggingface_hub already retries individual
# files/chunks on 429/5xx + resumes from cache; this wraps the WHOLE `hf download` so a sustained
# rate-limit or a transient that exhausts the library's budget gets a fresh, cache-resuming attempt.
# Tune with HF_DOWNLOAD_RETRIES (default 5). Backoff: 10s,20s,40s,… capped at 300s.
hf_retry() {
    local max="${HF_DOWNLOAD_RETRIES:-5}" n=1 wait=10 rc
    until "$@"; do
        rc=$?
        if [ "$n" -ge "$max" ]; then
            echo "[olmocore] download failed after $max attempts (exit $rc): $*" >&2
            return "$rc"
        fi
        echo "[olmocore] download attempt $n/$max failed (exit $rc); retrying in ${wait}s…" >&2
        sleep "$wait"
        n=$((n + 1)); wait=$((wait * 2))
        if [ "$wait" -gt 300 ]; then wait=300; fi
    done
}

# `hf download` coordinates its cache with filelock. A download hard-killed mid-setup (job/container torn
# down, node preempted) can leave an ORPHANED *.lock under <local-dir>/.cache/huggingface that a shared FS
# (Lustre) never releases — so the NEXT download deadlocks forever ("Still waiting to acquire lock on
# .gitignore.lock", Fetching 0/N). Only rank 0 downloads (gated below), so any lock present here has no live
# owner and is safe to clear before we start. Covers both the .gitignore.lock and the per-file .locks/*.lock.
clear_hf_locks() {
    local d="$1" n
    [ -n "$d" ] && [ -d "$d/.cache" ] || return 0
    n=$(find "$d/.cache" -name '*.lock' -type f 2>/dev/null | wc -l | tr -d ' ')
    [ "${n:-0}" -gt 0 ] || return 0
    echo "[olmocore] clearing $n stale hf lock(s) under $d/.cache (orphaned by a killed download)"
    find "$d/.cache" -name '*.lock' -type f -delete 2>/dev/null || true
}

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
        # The converter reads config.json via cached_path(), which treats a BARE HF repo id
        # (e.g. allenai/Olmo-3-7B-Think) as a LOCAL path -> FileNotFoundError. So if HF_MODEL
        # isn't already a local dir, stage it on the volume first; then both the config read
        # and the weight load resolve on-disk. hf download is idempotent/resumable.
        CONV_SRC="$HF_MODEL"
        if [ ! -d "$HF_MODEL" ]; then
            CONV_SRC="$DATA/hf_models/$HF_MODEL"
            echo "[olmocore] staging HF model $HF_MODEL -> $CONV_SRC"
            clear_hf_locks "$CONV_SRC"
            hf_retry hf download "$HF_MODEL" --local-dir "$CONV_SRC"
        fi
        # Attention sink: build the OLMo-core model WITH sinks so a sink-baked HF checkpoint
        # (self_attn.sinks) converts, and pre-fill sinks to OLMO_SINK_INIT for a stock warm-start.
        # Must match OLMO_USE_SINK at TRAIN time so the distcp <-> model params line up.
        SINK_CONV_ARGS=()
        [ "${OLMO_USE_SINK:-0}" = "1" ] && SINK_CONV_ARGS=(--use-sink --sink-init "${OLMO_SINK_INIT:-0.0}")
        python /workspace/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py \
            --checkpoint-input-path "$CONV_SRC" --model-arch "${MODEL_ARCH:-olmo3_7b}" \
            --tokenizer "$TOKENIZER_ARG" --output-dir "$CKPT_DIR" --skip-validation "${SINK_CONV_ARGS[@]}"
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
# GPU compute capability for arch gating below. NB: do NOT name this 'CC' — that's the reserved
# C-compiler env var (Triton's runtime/build.py and inductor read $CC as the compiler); if 'CC' is
# exported, a value like "9.0" leaks in and the build tries to exec a compiler named "9.0".
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
# FP8 scaling is arch-dependent (torch._scaled_mm via cuBLASLt; verified by tools/fp8_probe.py):
#   - sm_90 (Hopper) / sm_100 (DC Blackwell): ROWWISE works (accurate; per-row scale, DeepSeek recipe).
#   - sm_120 (RTX PRO 6000 workstation Blackwell): rowwise -> CUBLAS_STATUS_NOT_SUPPORTED, but
#     TENSORWISE works. Lower accuracy (one scale/tensor) — fine for a throughput A/B, NOT a shipped
#     model (do production FP8 on H200 with rowwise). Pick the default scaling by arch; explicit wins.
if [ "$PRECISION" = "fp8" ]; then
    case "$GPU_CC" in
        9.*|10.*) export OLMO_FP8="${OLMO_FP8:-rowwise}" ;;
        12.*)     export OLMO_FP8="${OLMO_FP8:-tensorwise}"
                  [ "$OLMO_FP8" = "rowwise" ] && { echo "ERROR: OLMO_FP8=rowwise unsupported on sm_120 (cuBLAS NOT_SUPPORTED); use tensorwise."; exit 2; }
                  echo "[olmocore] FP8 sm_120: tensorwise scaling (lower-accuracy, comparison-only; ship FP8 on H200/rowwise)" ;;
        *)        if [ -z "${OLMO_FORCE_FP8:-}" ]; then
                      echo "ERROR: PRECISION=fp8 on unrecognized GPU (cc $GPU_CC). FP8 verified on sm_90/sm_100 (rowwise)"
                      echo "       and sm_120 (tensorwise). Use bf16, or set OLMO_FP8=<tensorwise|rowwise> + OLMO_FORCE_FP8=1."
                      exit 2
                  fi
                  export OLMO_FP8="${OLMO_FP8:-tensorwise}" ;;
    esac
fi
case "$GPU_CC" in 9.*) DEFATTN=flash_3 ;; *) DEFATTN=flash_2 ;; esac
# FA3 is the DEFAULT on Hopper (sm_90 H100/H200) — including with attention sinks (OLMO_USE_SINK=1)
# and Ulysses CP, both of which FA3 supports (sink on the FA3 varlen path; ulysses via all-to-all).
# Force FA2 ONLY when RING CP will actually engage, since ring is FA2-only (FA3 raises on ring):
# that's a long sequence (seq_len > per-rank cap => CP engages) AND OLMO_CP_STYLE=ring (the default).
# Non-Hopper stays flash_2 (from the case above). An explicit OLMO_ATTN_BACKEND always wins below.
if [ "${SEQ_LEN:-65536}" -gt "${OLMO_MAX_TOKENS_PER_RANK:-16384}" ] && [ "${OLMO_CP_STYLE:-ring}" = "ring" ]; then
    DEFATTN=flash_2
fi
# 1b LOCAL TEST ONLY (does NOT affect 7b/32b): default to the SDPA/torch attention backend — it needs no
# special kernel, so it runs on ANY GPU incl. Ampere sm_86 (RTX 3090) where our FA2 build has no kernel.
# SDPA can't do intra-document masking, so also default OLMO_DOC_MASKING=0 (cross-doc attention — fine for
# a pipeline / write-correctness test). Override either env to force flash + doc-masking on a capable GPU.
if [ "$MODEL_SIZE" = "1b" ]; then
    DEFATTN=torch
    export OLMO_DOC_MASKING="${OLMO_DOC_MASKING:-0}"
fi
export OLMO_ATTN_BACKEND="${OLMO_ATTN_BACKEND:-$DEFATTN}"
export OLMO_SFT_SAVE_ROOT="${OLMO_SFT_SAVE_ROOT:-$DATA/checkpoints}"   # overridable: Fields train.py -> --output (OLMO_FP8 set arch-aware above; all-attn stays BF16)
# Shared filesystem across nodes (the default; our pipeline already assumes it — rank-0 stages on the
# shared mount + others wait on sentinels). Tells olmo-core get_fs_local_rank() = GLOBAL rank, so ONLY
# global-rank-0 does FS bookkeeping (dir/metadata/checkpoint-pruning/config). WITHOUT this, multi-node
# would have each node's local-rank-0 redundantly racing those ops on the shared dir. Set
# OLMO_SHARED_FS=0 only for a genuine per-node (non-shared) filesystem (then also set FS_LOCAL_RANK).
export OLMO_SHARED_FS="${OLMO_SHARED_FS:-1}"

# Optimizer: default SkipStepAdamW (the sft script pairs it with bf16 Adam moments) — spike protection
# (auto-skips a step when loss/grad-norm spikes past a rolling sigma band, logs `optim/step skipped`)
# AND ~16 GB/rank less optimizer VRAM than fused fp32 AdamW. Override with OLMO_OPTIM=fused_adamw (fp32
# fused-kernel baseline, fastest).
if [ -z "${OLMO_OPTIM:-}" ]; then
    OLMO_OPTIM=skip_step
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
        clear_hf_locks "$DL_DIR"
        hf_retry hf download "$DATASET_HF" "${HF_ARGS[@]}"
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
# Optional run-name suffix to distinguish launches in W&B + checkpoint dirs. Pass it EXPLICITLY so it's
# identical across nodes (the script does NOT auto-generate a timestamp: a per-node `date` would diverge
# across nodes and break the rendezvous id / save_folder). Single node: -e RUN_SUFFIX=$(date +%d%m%Y-%H%M)
# is evaluated once on the host so it's fine. Multi-node: set RUN_SUFFIX=$PBS_JOBID (or set RUN_NAME).
[ -n "${RUN_SUFFIX:-}" ] && RUN_NAME="${RUN_NAME}-${RUN_SUFFIX}"
echo "[olmocore] run: $RUN_NAME"
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

# ---- Multi-node launch (explicit contract; no inference) --------------------------------------
# Topology comes ONLY from the launcher's WORLD_SIZE (= number of NODES) and GLOBAL_RANK (= this
# node's 0-based index); both are required and already guarded at the top of this script (and, on the
# Fields path, validated in train.py). We do NOT auto-detect a process-level convention and there is
# NO standalone / localhost fallback — a misconfigured launch fails loud rather than fanning out into
# independent single-node jobs. torchrun spawns one rank per local GPU (--nproc_per_node) and assigns
# each child's RANK/LOCAL_RANK/WORLD_SIZE (the real process world_size = WORLD_SIZE_nodes x NPROC).
# A single-node run is just WORLD_SIZE=1/GLOBAL_RANK=0 with the node's own MASTER_ADDR.
NNODES="$WORLD_SIZE"
NODE_RANK="$GLOBAL_RANK"
RDZV=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
      --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
# torchrun assigns the children's ranks; drop the inherited node-level vars so they can't shadow the
# process-level RANK/WORLD_SIZE torchrun sets for each worker.
unset RANK WORLD_SIZE GLOBAL_RANK LOCAL_RANK 2>/dev/null || true

echo "[olmocore] $PRECISION | ${NNODES}x${NPROC} GPU cc=$GPU_CC | attn=$OLMO_ATTN_BACKEND | cp=${OLMO_CP_STYLE:-ring} | sink=${OLMO_USE_SINK:-0}${OLMO_USE_SINK:+@${OLMO_SINK_INIT:-0.0}} | ac=${OLMO_AC_BUDGET:-selected_ffn} | fp8=${OLMO_FP8:-off}${OLMO_FP8_FSDP_ALLGATHER:+/ag} | optim=$OLMO_OPTIM | ckpt=${OLMO_SAVE_INTERVAL:-1000}/${OLMO_EPHEMERAL_INTERVAL:-500}/keep${OLMO_KEEP_LAST_CKPTS} | node ${NODE_RANK}/${NNODES} | $DUR_VAL $DUR_UNIT"
# Size-specific SFT script (local copy of AI2's, beaker-stubbed). Resolved per MODEL_SIZE via the table
# above (DEF_SFT): 7b/32b -> Olmo-3-{7B,32B}-SFT-local.py (olmo3 long-context); 1b -> Olmo-2-1B-SFT-local.py
# (Olmo-2 1B local test path). Each size's MODEL_ARCH/HF_MODEL is set in the same table, so they stay in sync.
SFT_SCRIPT="$HERE/sft_scripts/$SFT_SCRIPT_NAME"
[ -f "$SFT_SCRIPT" ] || { echo "ERROR: no SFT script for MODEL_SIZE=$MODEL_SIZE at $SFT_SCRIPT"; exit 2; }

# Optional runtime pre-flight (OLMO_ATTN_SELFCHECK=1, rank-0 only, ~1 min): run the attention-sink
# kernel consistency checks on THIS box's GPU — eager vs FA2/FA3, in-kernel FA2 vs post-correction,
# and FA2-post-correction vs FA3-in-kernel (the two production paths). ABORTS the run if they diverge.
# Each check self-skips backends not present, so it validates whatever the image + GPU actually have.
if [ "${OLMO_ATTN_SELFCHECK:-0}" = "1" ] && [ "$THIS_NODE_RANK" -eq 0 ]; then
    echo "[olmocore] OLMO_ATTN_SELFCHECK=1: attention-sink kernel consistency check (FA2/FA3 vs eager, FA2 vs FA3)…"
    python /workspace/OLMo-core/src/test/nn/attention/attention_sink_flash_test.py \
        || { echo "[olmocore] ERROR: attention-sink self-check FAILED — aborting before training"; exit 7; }
fi

# HuggingFace checkpoint upload watchdog (node 0 only). If OLMO_HF_UPLOAD_REPO is set, a background poll
# loop converts each NEW complete distcp checkpoint -> HF (sink-preserving) and ships it to
# <repo>/step<N>/, marking each so it's shipped once. CPU-only (CUDA_VISIBLE_DEVICES="") so it never
# steals a training GPU. Needs HF_TOKEN with WRITE scope. Runtime-cloned (no image rebuild). After
# training, the watcher is killed and a FINAL uncapped upload lands the end-of-run model at the repo root.
UPLOAD_PY="$(dirname "$SFT_SCRIPT")/../upload.py"
UPLOAD_WATCHER_PID=""
# The to-HF converter needs a RESOLVABLE tokenizer repo (the bare "dolma2" alias is NOT one). This model
# (chankhavu/yccchen-olmo3-deploy) ships a DEEPSEEK tokenizer, NOT dolma2 — so default to the MODEL's own
# tokenizer: OLMO_TOKENIZER_HF (set when OLMO_HF_TOKENIZER=<id>/1), else HF_MODEL itself (carries the
# deepseek tokenizer). Never fall back to dolma2 here (would export a wrong tokenizer).
_UPLOAD_TOKENIZER="${OLMO_TOKENIZER_HF:-$HF_MODEL}"
# Upload path prefix inside the repo. Default = a run-unique name so re-runs don't overwrite each other
# (like the FP8/NII run): <base>-<YYYYMMDDHHMMSS>. Computed ONCE here (NOT per poll) so every checkpoint
# of THIS run lands under the same prefix. An explicit OLMO_HF_UPLOAD_PREFIX is used verbatim (then you own
# uniqueness). Passed to both the watcher and the final upload so they agree.
_UPLOAD_PREFIX="${OLMO_HF_UPLOAD_PREFIX:-ycchen-olmo32b-ds-sft-$(date +%Y%m%d%H%M%S)}"
_reap_watcher() { [ -n "$UPLOAD_WATCHER_PID" ] && kill -TERM "$UPLOAD_WATCHER_PID" 2>/dev/null || true; }
if [ -n "${OLMO_HF_UPLOAD_REPO:-}" ] && [ "$THIS_NODE_RANK" -eq 0 ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[olmocore] WARN: OLMO_HF_UPLOAD_REPO set but HF_TOKEN is empty — skipping upload watcher."
    elif [ ! -f "$UPLOAD_PY" ]; then
        echo "[olmocore] WARN: upload.py not found at $UPLOAD_PY — skipping upload watcher."
    else
        UPLOAD_LOG="${OLMO_SFT_SAVE_ROOT}/hf_upload_watcher.log"
        echo "[olmocore] HF upload watcher -> ${OLMO_HF_UPLOAD_REPO} (poll ${OLMO_HF_UPLOAD_INTERVAL:-300}s, log $UPLOAD_LOG)"
        CUDA_VISIBLE_DEVICES="" nohup python "$UPLOAD_PY" --watch \
            --output "$OLMO_SFT_SAVE_ROOT" --repo "$OLMO_HF_UPLOAD_REPO" --run-name "$RUN_NAME" \
            --prefix "$_UPLOAD_PREFIX" \
            --seq-len "${SEQ_LEN:-65536}" --tokenizer "$_UPLOAD_TOKENIZER" \
            > "$UPLOAD_LOG" 2>&1 &
        UPLOAD_WATCHER_PID=$!
    fi
fi

# Run torchrun in the BACKGROUND and FORWARD preemption signals to it. Old code `exec`'d torchrun as PID 1,
# so a Beaker/docker SIGTERM hit it directly (graceful checkpoint). Now run.sh is PID 1, so we must trap
# TERM/INT, forward to torchrun, and RE-WAIT until it finishes its shutdown save — else it's SIGKILLed
# mid-save = lost progress. The watcher traps SIGTERM and reaps its own in-flight convert.
TRAIN_RC=0
trap '_reap_watcher; [ -n "${TORCHRUN_PID:-}" ] && kill -TERM "$TORCHRUN_PID" 2>/dev/null || true' TERM INT
trap '_reap_watcher' EXIT
torchrun "${RDZV[@]}" --nproc_per_node="$NPROC" \
    "$SFT_SCRIPT" \
    train "$RUN_NAME" "$CKPT" "${CLUSTER:-local_h100}" \
    --seq_len="${SEQ_LEN:-65536}" --num_nodes="$NNODES" \
    --global_batch_size="${GLOBAL_BATCH_SIZE:-$DEF_GBS}" \
    --dataset_path="$DATASET" \
    --train_module.optim.lr="${LR:-$DEF_LR}" \
    --trainer.max_duration.value="$DUR_VAL" --trainer.max_duration.unit="$DUR_UNIT" \
    ${EXTRA[@]+"${EXTRA[@]}"} &
TORCHRUN_PID=$!
while :; do
    if wait "$TORCHRUN_PID"; then TRAIN_RC=0; else TRAIN_RC=$?; fi
    kill -0 "$TORCHRUN_PID" 2>/dev/null || break   # a forwarded signal interrupts wait; re-wait until it exits
done
trap - TERM INT

# Stop the watcher (it reaps any in-flight convert), then the FINAL uncapped upload (node 0). Runs on BOTH
# a normal end AND a crash — on a non-zero rc it ships the LATEST SURVIVABLE complete checkpoint
# (find_final_checkpoint skips half-written ones), matching the NII run. So the three upload triggers are:
# (1) each new checkpoint mid-run (the watcher), (2) crash, (3) normal end. Repo root (or <prefix>) = model.
_reap_watcher
if [ -n "${OLMO_HF_UPLOAD_REPO:-}" ] && [ "$THIS_NODE_RANK" -eq 0 ] && [ -n "${HF_TOKEN:-}" ] && [ -f "$UPLOAD_PY" ]; then
    if [ "$TRAIN_RC" -eq 0 ]; then
        echo "[olmocore] final HF upload -> ${OLMO_HF_UPLOAD_REPO} (end-of-run model)"
    else
        echo "[olmocore] training exited rc=$TRAIN_RC — shipping the LATEST SURVIVABLE checkpoint -> ${OLMO_HF_UPLOAD_REPO}"
    fi
    CUDA_VISIBLE_DEVICES="" python "$UPLOAD_PY" --final \
        --output "$OLMO_SFT_SAVE_ROOT" --repo "$OLMO_HF_UPLOAD_REPO" --run-name "$RUN_NAME" \
        --prefix "$_UPLOAD_PREFIX" \
        --seq-len "${SEQ_LEN:-65536}" --tokenizer "$_UPLOAD_TOKENIZER" \
        || echo "[olmocore] WARN: final HF upload failed — the distcp checkpoint is still on disk/WEKA."
fi
exit "$TRAIN_RC"
