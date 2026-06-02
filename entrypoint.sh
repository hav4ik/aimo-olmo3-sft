#!/bin/bash
# Repo entrypoint (run by bootstrap.sh after the code is cloned). Selects the framework
# and runs training. With no FRAMEWORK set, drops to a shell — handy for first-run poking.
# Env contract (all optional except a FRAMEWORK or EXPERIMENT to actually train):
#   EXPERIMENT = <size>_<precision>_<variant>  e.g. 32b_fp8_cot — ONE knob that sets MODEL_SIZE,
#                PRECISION, DATASET_NAME, DATASET_HF (see experiments.sh). Defaults FRAMEWORK=olmocore.
#   FRAMEWORK = olmocore | axolotl
#   MODEL_SIZE = 7b | 32b             (olmocore; default 7b)
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

# Locate ourselves so the code runs whether it was git-cloned (/workspace/code) or baked (/opt/aimo-code).
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# One-knob front-end: EXPERIMENT=<size>_<precision>_<variant> -> MODEL_SIZE/PRECISION/DATASET_NAME/
# DATASET_HF (see experiments.sh), so the launch is just the topology env + `-e EXPERIMENT=...`.
# Explicit env vars still override. Defaults FRAMEWORK to olmocore when an EXPERIMENT is given.
if [ -n "${EXPERIMENT:-}" ]; then
    source "$CODE_ROOT/experiments.sh"
    resolve_experiment "$EXPERIMENT" || exit 2
fi
echo "[entrypoint] EXPERIMENT=${EXPERIMENT:-<none>} FRAMEWORK=${FRAMEWORK:-olmocore} MODEL_SIZE=${MODEL_SIZE:-7b} PRECISION=${PRECISION:-bf16} STAGE=${STAGE:-train}"

DEFAULT_FW=""; [ -n "${EXPERIMENT:-}" ] && DEFAULT_FW=olmocore
case "${FRAMEWORK:-$DEFAULT_FW}" in
    olmocore) exec bash "$CODE_ROOT/olmocore/run.sh" "$@" ;;
    axolotl)  exec bash "$CODE_ROOT/axolotl/run.sh"  "$@" ;;
    "") echo "[entrypoint] No EXPERIMENT/FRAMEWORK set -> shell. Set EXPERIMENT=<size>_<prec>_<variant> or FRAMEWORK=olmocore|axolotl."; exec bash ;;
    *)  echo "[entrypoint] Unknown FRAMEWORK='${FRAMEWORK}' (want olmocore|axolotl)"; exit 2 ;;
esac
