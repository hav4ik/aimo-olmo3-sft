#!/bin/bash
# Repo entrypoint (run by bootstrap.sh after the code is cloned). Selects the framework
# and runs training. With no FRAMEWORK set, drops to a shell — handy for first-run poking.
# Env contract (all optional except FRAMEWORK to actually train):
#   FRAMEWORK = olmocore | axolotl
#   PRECISION = bf16 | fp8            (default bf16)
#   STAGE     = train | convert       (olmocore only; convert = HF->OLMo-core checkpoint)
#   NPROC_PER_NODE                     (default = all visible GPUs)
#   HF_TOKEN, WANDB_API_KEY            (passed by `docker run -e`; never baked)
# Data + checkpoints live under /data/training (mounted on the node).
set -euo pipefail
export HF_HOME="${HF_HOME:-/data/training/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-/data/training/wandb}"
mkdir -p /data/training/checkpoints "$HF_HOME" "$WANDB_DIR"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[entrypoint] GPUs: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | paste -sd'; ' -)"
echo "[entrypoint] FRAMEWORK=${FRAMEWORK:-<none>} PRECISION=${PRECISION:-bf16} STAGE=${STAGE:-train}"

case "${FRAMEWORK:-}" in
    olmocore) exec bash /workspace/code/olmocore/run.sh "$@" ;;
    axolotl)  exec bash /workspace/code/axolotl/run.sh  "$@" ;;
    "") echo "[entrypoint] No FRAMEWORK set -> shell. Set FRAMEWORK=olmocore|axolotl (+ PRECISION) to train."; exec bash ;;
    *)  echo "[entrypoint] Unknown FRAMEWORK='${FRAMEWORK}' (want olmocore|axolotl)"; exit 2 ;;
esac
