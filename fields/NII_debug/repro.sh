#!/usr/bin/env bash
# Reproduce the NII run_01 crash locally: a non-executable `nvcc` on PATH ->
# torch.compile -> PermissionError [Errno 13] Permission denied: 'nvcc'.
#
# Two levels:
#   ./repro.sh min      -> exact exception, NO GPU needed (stdlib only)
#   ./repro.sh full     -> full InductorError traceback (needs torch + a CUDA GPU)
#
# Run `full` either bare-metal on the 3090 box, or inside the shipped image:
#   singularity exec --nv olmo-sft-v2-allsm.sif bash fields/NII_debug/repro.sh full
set -euo pipefail
MODE="${1:-min}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# Stage the NII condition: an nvcc that is present but NOT executable, and make it
# the only nvcc on PATH (strip any real cuda dir so there's no working fallback).
printf '#!/bin/sh\necho "fake nvcc"\n' > "$WORK/nvcc"
chmod 644 "$WORK/nvcc"                       # readable, no +x  == noexec / missing-x on the host
export PATH="$WORK:$(echo "$PATH" | tr ':' '\n' | grep -v -i cuda | paste -sd:)"
echo "staged non-exec nvcc: $(ls -l "$WORK/nvcc" | awk '{print $1, $NF}')"
echo "nvcc resolves to    : $(command -v nvcc || echo '<none>')"

if [ "$MODE" = "min" ]; then
  python3 - <<'PY'
import subprocess
try:
    subprocess.check_output(["nvcc", "--version"])
except Exception as e:
    print(f"REPRODUCED: {type(e).__name__}: {e}")
PY
else
  python3 "$(dirname "$0")/compile_test.py"
fi
