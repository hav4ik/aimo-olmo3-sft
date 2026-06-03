#!/bin/bash
# ENTRYPOINT. Clones THIS repo's code at container start, so the code iterates independently of the
# heavy-deps image (we change configs/recipes far more often than the image). SINGLE source of
# truth = the git ref you ask for. Pin a run with CODE_REF=<branch|tag|commit> (default: the
# olmo3-sft branch); the resolved commit SHA is printed at startup so any run is traceable/debuggable.
#
# Clones into the MOUNTED run storage (CODE_DIR, default /data/training/code) — writable even under a
# read-only Singularity rootfs, and the exact code that ran sits next to the checkpoints. Multi-node:
# node-rank 0 stages the code on the shared storage and the other nodes use it (no clone race,
# identical code on every rank). No secrets here; HF_TOKEN/WANDB_API_KEY come from the container env.
set -euo pipefail
REPO="${CODE_REPO:-https://github.com/hav4ik/aimo-olmo3-sft}"
REF="${CODE_REF:-${CODE_BRANCH:-olmo3-sft}}"
DEST="${CODE_DIR:-/data/training/code}"
READY="$DEST/.code_ready"
RANK="${NODE_RANK:-${GLOBAL_RANK:-0}}"

# Mirror ALL output (this script + the exec'd entrypoint/run.sh/torchrun) to a per-rank log file
# on the writable run storage, so you can `tail -f` it from a SEPARATE shell into the container
# (the launching session is often a Jupyter terminal a remote shell can't see). Fixed location,
# no knob. Guarded so a logging failure never aborts the run.
LOG_DIR=/data/training/logs
if mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG_FILE="$LOG_DIR/run-rank${RANK}.log"
    echo "[bootstrap] logging to $LOG_FILE  (tail -f it from another shell)"
    exec > >(tee -a "$LOG_FILE") 2>&1
else
    echo "[bootstrap] WARN: $LOG_DIR not writable; console-only logging"
fi

# Early W&B log streaming (preflight visibility): get clone/convert/download/CP-init failures onto
# W&B BEFORE training's own run exists. A background helper opens a "<group>-preflight" run in the
# same project+group and live-syncs $LOG_FILE; it self-finishes when bootstrap's exec'd process tree
# exits (survives the final `exec` by watching its parent pid). Rank 0 only; needs WANDB_API_KEY and
# an importable wandb (the olmocore conda base has it; silently skipped otherwise, e.g. axolotl venv).
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXPERIMENT:-olmo3-sft}}"   # also groups the training run
    export WANDB_DIR="${WANDB_DIR:-/data/training/wandb}"; mkdir -p "$WANDB_DIR" 2>/dev/null || true
    if [ "$RANK" -eq 0 ] && [ -n "${LOG_FILE:-}" ] && python -c "import wandb" 2>/dev/null; then
        WANDB_PREFLIGHT_LOG="$LOG_FILE" nohup python - >/dev/null 2>&1 <<'PY' &
import os, time, wandb
logf = os.environ["WANDB_PREFLIGHT_LOG"]
ppid0 = os.getppid()
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "olmo3-7b-sft"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    group=os.environ.get("WANDB_RUN_GROUP"),
    name=(os.environ.get("WANDB_RUN_GROUP") or "job") + "-preflight",
    job_type="preflight",
)
for _ in range(600):                 # wait for the log file to appear
    if os.path.exists(logf):
        break
    time.sleep(1)
wandb.save(logf, policy="live")      # live-sync to the run's Files tab as it grows
while os.getppid() == ppid0:         # stay alive until the exec'd job tree exits
    time.sleep(15)
run.finish()
PY
        echo "[bootstrap] streaming $LOG_FILE -> W&B (${WANDB_RUN_GROUP}-preflight)"
    fi
fi

if [ "$RANK" -eq 0 ]; then
    rm -f "$READY"; mkdir -p "$DEST"
    if [ -d "$DEST/.git" ]; then
        echo "[bootstrap] updating $DEST from $REPO"
        git -C "$DEST" remote set-url origin "$REPO"
        git -C "$DEST" fetch --all --tags --prune
    else
        echo "[bootstrap] cloning $REPO -> $DEST"
        git clone "$REPO" "$DEST"
    fi
    git -C "$DEST" checkout -f "$REF"
    git -C "$DEST" reset --hard "origin/$REF" 2>/dev/null || true   # fast-forward if REF is a branch
    SHA="$(git -C "$DEST" rev-parse --short HEAD)"
    echo "[bootstrap] code: ${REF} @ ${SHA} — $(git -C "$DEST" log -1 --pretty=%s)"
    echo "$SHA" > "$READY"
else
    echo "[bootstrap] node_rank=$RANK waiting for rank-0 to stage code at $DEST ..."
    sleep 5   # let rank-0 clear any stale sentinel from a previous run first
    for _ in $(seq 1 180); do [ -f "$READY" ] && break; sleep 5; done
    [ -f "$READY" ] || { echo "[bootstrap] ERROR: timed out waiting for code at $READY"; exit 5; }
    echo "[bootstrap] using rank-0 staged code @ $(cat "$READY")"
fi
exec bash "$DEST/entrypoint.sh" "$@"
