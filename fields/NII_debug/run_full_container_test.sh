#!/usr/bin/env bash
# ============================================================================
# run_full_container_test.sh — run the ACTUAL container entrypoint (train.py)
# end-to-end, with a NON-EXECUTABLE nvcc bound at the NII vector, and confirm it
# passes the dry-run compile (where the NII run crashed) and reaches training steps.
#
# This is the full-fat version of the nvcc fix check: it boots the real training
# (download/convert -> build -> torch.compile at the dry-run -> training steps),
# not a minimal torch.compile probe. Use a short --max-steps so it exits quickly.
#
# Defaults run the local .sif on this box, reusing the cached model/dataset under
# $SCRATCH. Override via env:
#   SIF=...            path to the .sif (secrets baked)            [default: repo/olmo-sft-v2.1-allsm.sif]
#   SCRATCH=...        host dir bound to /tmp (needs ~30GB+ free)  [default: /mnt/data/sif-test/tmp]
#   INJECT_BROKEN_NVCC=1|0   plant a non-exec nvcc at the NII vector to prove robustness  [default: 1]
#   MAX_STEPS=...      training steps before exit                  [default: 10]
#   EXPERIMENT=...     recipe                                      [default: olmo_1b_bf16]
#
# Needs a GPU (uses --nv). The .sif already bakes HF/W&B creds, so no tokens needed.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SIF="${SIF:-$REPO/olmo-sft-v2.1-allsm.sif}"
SCRATCH="${SCRATCH:-/mnt/data/sif-test/tmp}"
INJECT_BROKEN_NVCC="${INJECT_BROKEN_NVCC:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
EXPERIMENT="${EXPERIMENT:-olmo_1b_bf16}"
RUN_SUFFIX="${RUN_SUFFIX:-fulltest}"
LOG="${LOG:-$REPO/fields/NII_debug/full_container_test.log}"
VECTOR="/usr/local/cuda/bin/nvcc"   # where a host nvcc would land on the container PATH

[ -f "$SIF" ] || { echo "ERROR: SIF not found: $SIF (set SIF=...)"; exit 2; }
mkdir -p "$SCRATCH"

BINDS=(--bind "$SCRATCH:/tmp")
TMP=""
if [ "$INJECT_BROKEN_NVCC" = 1 ]; then
  TMP="$(mktemp -d)"; printf '#!/bin/sh\necho "fake host nvcc"\n' > "$TMP/nvcc"; chmod 0644 "$TMP/nvcc"
  BINDS+=(--bind "$TMP/nvcc:$VECTOR")
  echo ">> injecting a NON-EXECUTABLE nvcc at $VECTOR (simulates the NII host noexec nvcc)"
fi
trap '[ -n "$TMP" ] && rm -rf "$TMP"' EXIT

echo ">> running the container ENTRYPOINT (python /app/train.py) — $EXPERIMENT, $MAX_STEPS steps, COLD compile"
echo ">> log: $LOG"
echo "============================================================"
# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 forces a cold compile so the nvcc path actually fires.
singularity run --nv --containall "${BINDS[@]}" \
  --env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  "$SIF" \
  --experiment "$EXPERIMENT" --seq-len 8192 --run-suffix "$RUN_SUFFIX" \
  --max-steps "$MAX_STEPS" --save-interval 1000 --ephemeral-interval 500 \
  --learning_rate 1e-12 --global-batch-tokens 16384 --no-remote-shell 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

echo "============================================================"
if grep -qE "Permission denied: 'nvcc'|InductorError: PermissionError" "$LOG"; then
  echo "RESULT: FAIL — the nvcc crash occurred (this is the bug)."; exit 1
elif grep -qE "step=[0-9]" "$LOG"; then
  echo "RESULT: PASS — entrypoint passed the dry-run compile and reached training steps (container exit rc=$RC)."; exit 0
else
  echo "RESULT: INCONCLUSIVE (container rc=$RC) — no nvcc error, but no training step seen. Check $LOG."; exit 3
fi
