#!/bin/bash
# Repo entrypoint (run by bootstrap.sh after the code is cloned). Selects the framework
# and runs training. With no FRAMEWORK set, drops to a shell — handy for first-run poking.
# Env contract (all optional except a FRAMEWORK or EXPERIMENT to actually train):
#   EXPERIMENT = <size>_<precision>_<variant>  e.g. 32b_fp8_cot — ONE knob that sets MODEL_SIZE,
#                PRECISION, DATASET_NAME, DATASET_HF (see experiments.sh).
#   FRAMEWORK = olmocore | axolotl    (rarely needed — defaults to the IMAGE's baked SFT_FRAMEWORK)
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
# Framework = the IMAGE's baked identity (SFT_FRAMEWORK, set in each deploy Dockerfile), so you never
# pass FRAMEWORK — the olmocore image runs olmocore, the axolotl image runs axolotl. An explicit
# FRAMEWORK env still wins. Back-compat: images built before the marker fall back to olmocore.
FW="${FRAMEWORK:-${SFT_FRAMEWORK:-}}"
[ -z "$FW" ] && [ -n "${EXPERIMENT:-}" ] && FW=olmocore
echo "[entrypoint] EXPERIMENT=${EXPERIMENT:-<none>} FRAMEWORK=${FW:-<none>} MODEL_SIZE=${MODEL_SIZE:-7b} PRECISION=${PRECISION:-bf16} STAGE=${STAGE:-train}"

case "$FW" in
    olmocore) exec bash "$CODE_ROOT/olmocore/run.sh" "$@" ;;
    axolotl)  exec bash "$CODE_ROOT/axolotl/run.sh"  "$@" ;;
    "") echo "[entrypoint] No framework -> shell. (Image should bake SFT_FRAMEWORK; or set FRAMEWORK=olmocore|axolotl.)"; exec bash ;;
    *)  echo "[entrypoint] Unknown FRAMEWORK='${FW}' (want olmocore|axolotl)"; exit 2 ;;
esac
