#!/bin/bash
# Host-side launcher for the olmo-core SFT container: turns the long `docker run -e …` into named
# flags with sensible defaults. Runs ONE container that (on a fresh data dir) converts the HF model
# -> distcp and then trains — same single-command flow as the NII singularity runs.
#
# Quick start (128-token smoke on 8x H100, current model/dataset are the defaults):
#   ./olmocore/launch.sh --data /DATA/olmo-run --seq-len 128 --max-steps 10 \
#       --gbs 1024 --max-tokens-per-rank 128
#
# Real run (2 epochs, drop the smoke overrides):
#   ./olmocore/launch.sh --data /DATA/olmo-run --seq-len 65536 --epochs 2 --cp-style ulysses
#
# HF_TOKEN / WANDB_API_KEY are read from your shell env and forwarded if set.
# Use --dry-run to print the docker command without running it. --help for all flags.
set -euo pipefail

# ---- defaults (the current yccchen sink run; override any with a flag) ------------------------
IMAGE="chankhavu/olmo3-olmocore:cu128-fa2-sink"
CODE_REF="olmocore-cu128-fa2-sink"
MODEL_SIZE="32b"
SFT_SCRIPT="Olmo-3-32B-SFT-bf16.py"
HF_MODEL="chankhavu/yccchen-olmo3-deploy"
DATASET_HF="chankhavu/yccchen-stage2-olmocore-256k-v2"
DATASET_SUBDIR=""              # this dataset keeps its shards at the repo root
DATASET_NAME="yccchen-stage2"
DATA=""                        # REQUIRED: host dir bind-mounted to /data/training

SEQ_LEN="65536"
EPOCHS="2"
MAX_STEPS=""                   # if set, overrides epochs (steps mode) — for smokes
GLOBAL_BATCH_SIZE=""           # empty => the script's per-size default
MAX_TOKENS_PER_RANK=""         # empty => script default (16384); drives cp_degree
CP_STYLE=""                    # empty => ring; or ulysses
LR=""

USE_SINK="1"                   # this model is Olmo3SinkForCausalLM
HF_TOKENIZER="1"               # 1 => use HF_MODEL's own tokenizer (vocab+eos/pad/bos+YaRN); or an id
SINK_INIT=""                   # only for a stock (no-sink) warm start; sink-baked ckpt ignores it
STAGE=""                       # empty => train (auto-converts first); or 'convert' to only convert

# distributed: single node by default; a multi-node launcher sets these per node
NNODES="1"; NODE_RANK="0"; MASTER_ADDR="127.0.0.1"; MASTER_PORT="29400"
GPUS="all"
DRY_RUN=0
declare -a EXTRA_ENV=()        # --env KEY=VAL (repeatable) for anything not covered

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Flags (defaults in []):
  --data DIR            host dir -> /data/training (REQUIRED) [${DATA:-<unset>}]
  --image REF           container image [$IMAGE]
  --code-ref REF        aimo-olmo3-sft branch/tag/sha cloned at runtime [$CODE_REF]
  --model ID            HF base model (HF_MODEL) [$HF_MODEL]
  --dataset ID          HF tokenized dataset (DATASET_HF) [$DATASET_HF]
  --dataset-subdir SUB  subdir with token_ids_part_*.npy ("" = repo root) [${DATASET_SUBDIR:-<root>}]
  --dataset-name NAME   local dir name under /data/training/datasets [$DATASET_NAME]
  --model-size 7b|32b   [$MODEL_SIZE]         --sft-script NAME   [$SFT_SCRIPT]
  --seq-len N           [$SEQ_LEN]            --epochs N          [$EPOCHS]
  --max-steps N         steps mode (overrides epochs) [${MAX_STEPS:-off}]
  --gbs N               global batch tokens [${GLOBAL_BATCH_SIZE:-per-size default}]
  --max-tokens-per-rank N   per-rank cap that sets cp_degree [${MAX_TOKENS_PER_RANK:-16384}]
  --cp-style ring|ulysses   [${CP_STYLE:-ring}]      --lr LR   [${LR:-per-size default}]
  --sink 0|1            per-head attention sink [$USE_SINK]
  --hf-tokenizer 0|1|ID use the model's own tokenizer + YaRN [$HF_TOKENIZER]
  --sink-init F         initial sink logit (stock warm start only) [${SINK_INIT:-model/0.0}]
  --stage train|convert [${STAGE:-train}]
  --nnodes N            number of NODES (WORLD_SIZE) [$NNODES]
  --node-rank N         this node's index (GLOBAL_RANK) [$NODE_RANK]
  --master-addr HOST    rendezvous host [$MASTER_ADDR]      --master-port P [$MASTER_PORT]
  --gpus SPEC           docker --gpus value [$GPUS]
  --env KEY=VAL         extra env (repeatable)
  --dry-run             print the docker command, don't run
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        --code-ref) CODE_REF="$2"; shift 2;;
        --model) HF_MODEL="$2"; shift 2;;
        --dataset) DATASET_HF="$2"; shift 2;;
        --dataset-subdir) DATASET_SUBDIR="$2"; shift 2;;
        --dataset-name) DATASET_NAME="$2"; shift 2;;
        --model-size) MODEL_SIZE="$2"; shift 2;;
        --sft-script) SFT_SCRIPT="$2"; shift 2;;
        --seq-len) SEQ_LEN="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --max-steps) MAX_STEPS="$2"; shift 2;;
        --gbs) GLOBAL_BATCH_SIZE="$2"; shift 2;;
        --max-tokens-per-rank) MAX_TOKENS_PER_RANK="$2"; shift 2;;
        --cp-style) CP_STYLE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --sink) USE_SINK="$2"; shift 2;;
        --hf-tokenizer) HF_TOKENIZER="$2"; shift 2;;
        --sink-init) SINK_INIT="$2"; shift 2;;
        --stage) STAGE="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --node-rank) NODE_RANK="$2"; shift 2;;
        --master-addr) MASTER_ADDR="$2"; shift 2;;
        --master-port) MASTER_PORT="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --env) EXTRA_ENV+=("$2"); shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2;;
    esac
done

[ -n "$DATA" ] || { echo "ERROR: --data DIR is required" >&2; exit 2; }

# ---- assemble the container env ---------------------------------------------------------------
declare -a ENVS=(
    -e "WORLD_SIZE=$NNODES" -e "GLOBAL_RANK=$NODE_RANK"
    -e "MASTER_ADDR=$MASTER_ADDR" -e "MASTER_PORT=$MASTER_PORT"
    -e "CODE_REF=$CODE_REF"
    -e "MODEL_SIZE=$MODEL_SIZE" -e "SFT_SCRIPT_NAME=$SFT_SCRIPT"
    -e "HF_MODEL=$HF_MODEL"
    -e "DATASET_HF=$DATASET_HF" -e "DATASET_SUBDIR=$DATASET_SUBDIR" -e "DATASET_NAME=$DATASET_NAME"
    -e "SEQ_LEN=$SEQ_LEN"
    -e "OLMO_USE_SINK=$USE_SINK"
)
# HF tokenizer: 1 => reuse HF_MODEL; an id => that model's tokenizer; 0/empty => dolma2 default
[ "$HF_TOKENIZER" != "0" ] && [ -n "$HF_TOKENIZER" ] && ENVS+=(-e "OLMO_HF_TOKENIZER=$HF_TOKENIZER")
[ -n "$STAGE" ]               && ENVS+=(-e "STAGE=$STAGE")
[ -n "$MAX_STEPS" ]           && ENVS+=(-e "MAX_STEPS=$MAX_STEPS")   || ENVS+=(-e "EPOCHS=$EPOCHS")
[ -n "$GLOBAL_BATCH_SIZE" ]   && ENVS+=(-e "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE")
[ -n "$MAX_TOKENS_PER_RANK" ] && ENVS+=(-e "OLMO_MAX_TOKENS_PER_RANK=$MAX_TOKENS_PER_RANK")
[ -n "$CP_STYLE" ]            && ENVS+=(-e "OLMO_CP_STYLE=$CP_STYLE")
[ -n "$LR" ]                  && ENVS+=(-e "LR=$LR")
[ -n "$SINK_INIT" ]          && ENVS+=(-e "OLMO_SINK_INIT=$SINK_INIT")
# forward secrets from the host env only if present
[ -n "${HF_TOKEN:-}" ]       && ENVS+=(-e "HF_TOKEN=$HF_TOKEN")
[ -n "${WANDB_API_KEY:-}" ]  && ENVS+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
for kv in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do ENVS+=(-e "$kv"); done

set -- docker run --rm --gpus "$GPUS" --ipc=host \
    -v "$DATA:/data/training" \
    "${ENVS[@]}" \
    "$IMAGE"

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"; echo
else
    echo "[launch] node ${NODE_RANK}/${NNODES} | $HF_MODEL | seq=$SEQ_LEN | ${MAX_STEPS:+steps=$MAX_STEPS}${MAX_STEPS:-epochs=$EPOCHS} | sink=$USE_SINK"
    exec "$@"
fi
