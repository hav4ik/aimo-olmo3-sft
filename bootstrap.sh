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

# --- Beaker multi-node rendezvous shim --------------------------------------------------------------
# run.sh (further down the chain) HARD-REQUIRES WORLD_SIZE(#nodes) / GLOBAL_RANK(node-index) /
# MASTER_ADDR / MASTER_PORT. Launched via train.py or the host launch.sh, those are set explicitly. But
# AI2 Beaker runs this ENTRYPOINT directly and injects rendezvous info under BEAKER_* names instead, so
# map them here — BEFORE the rank-0 code-staging check below (else every replica thinks it is rank 0 and
# they race to clone into the shared FS). Explicit env always wins (:- only fills unset vars); non-Beaker
# launches are untouched (run.sh keeps its fail-loud check when nothing set the topology).
#   BEAKER_REPLICA_COUNT -> WORLD_SIZE   BEAKER_REPLICA_RANK -> GLOBAL_RANK
#   BEAKER_LEADER_REPLICA_HOSTNAME -> MASTER_ADDR (needs leaderSelection+replicas>1+hostNetworking)
if [ -n "${BEAKER_REPLICA_COUNT:-}" ]; then
    if [ "${BEAKER_REPLICA_COUNT}" -gt 1 ] && [ -z "${BEAKER_LEADER_REPLICA_HOSTNAME:-}" ] && [ -z "${MASTER_ADDR:-}" ]; then
        echo "[bootstrap] WARN: BEAKER_REPLICA_COUNT=$BEAKER_REPLICA_COUNT but BEAKER_LEADER_REPLICA_HOSTNAME is unset — set leaderSelection:true + hostNetworking:true in the Beaker spec, or torchrun can't find the master node."
    fi
    export WORLD_SIZE="${WORLD_SIZE:-$BEAKER_REPLICA_COUNT}"
    export GLOBAL_RANK="${GLOBAL_RANK:-${BEAKER_REPLICA_RANK:-0}}"
    export MASTER_ADDR="${MASTER_ADDR:-${BEAKER_LEADER_REPLICA_HOSTNAME:-127.0.0.1}}"
    export MASTER_PORT="${MASTER_PORT:-29400}"
    echo "[bootstrap] Beaker rendezvous (BEAKER_REPLICA_*): WORLD_SIZE=$WORLD_SIZE GLOBAL_RANK=$GLOBAL_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
elif env | grep -q '^BEAKER_' && [ -z "${WORLD_SIZE:-}" ]; then
    # Single-replica Beaker job (no BEAKER_REPLICA_*): default to a single node.
    export WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400
    echo "[bootstrap] Beaker single-node (no BEAKER_REPLICA_*): WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400"
fi

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
