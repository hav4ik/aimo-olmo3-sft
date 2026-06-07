#!/usr/bin/env bash
# ============================================================================
# nvcc_selfcheck.sh — diagnose & test the "host nvcc breaks torch.compile" bug
# ============================================================================
# THE FAILURE CLASS (for cluster admins)
#   A container that ships no nvcc of its own, run under Singularity/Apptainer,
#   can resolve a HOST nvcc through the inherited PATH (or a host CUDA dir bound
#   onto an on-PATH location such as /usr/local/cuda/bin). If that host nvcc sits
#   on a `noexec` mount (standard security hardening) or lacks +x for the job
#   user, PyTorch's torch.compile crashes EVERY rank at the first (cold) compile:
#
#       torch._inductor.exc.InductorError:
#           PermissionError: [Errno 13] Permission denied: 'nvcc'
#
#   PyTorch tolerates a *missing* nvcc (it catches FileNotFoundError and prints
#   "# nvcc not found") but NOT a *non-executable* one (PermissionError is
#   uncaught) — so "present but not executable" is the one fatal state. It only
#   fires on a COLD inductor cache (a warm cache skips codegen), so it looks like
#   it hits every fresh run until a cache exists.
#
# WHAT THIS SCRIPT DOES
#   diagnose <image>  — show, INSIDE the container, which nvcc resolves, whether
#                       it's a host path, its perms/owner, and its mount options
#                       (is it noexec?). Run this on your cluster to see if your
#                       containers would resolve a non-executable host nvcc.
#   test <image>      — plant a deliberately NON-EXECUTABLE nvcc on the container
#                       PATH and run a cold torch.compile. PASS = the container
#                       survived (it ships/uses its own nvcc, or none); FAIL = it
#                       crashed with the PermissionError above (vulnerable image).
#
# THE TWO FIXES
#   (image side) ship an executable nvcc first on PATH so the host's is never
#               used — what this project's v2.1 image does (/opt/fields/bin/nvcc).
#   (host side)  ensure nvcc is either absent or executable for the job — e.g.
#               don't bind a host CUDA toolkit onto the container PATH, or mount
#               it exec, or drop /usr/local/cuda/bin from the container PATH.
#
# USAGE
#   ./nvcc_selfcheck.sh diagnose /path/to/image.sif
#   ./nvcc_selfcheck.sh test     /path/to/image.sif
#   RUNNER=docker ./nvcc_selfcheck.sh test  repo/image:tag     # use docker instead
#
# Requires a GPU + the GPU runtime flag (--nv for singularity, --gpus all for docker).
# ============================================================================
set -uo pipefail

MODE="${1:-}"; IMAGE="${2:-}"
RUNNER="${RUNNER:-singularity}"            # singularity | docker
VECTOR="${VECTOR:-/usr/local/cuda/bin/nvcc}"   # where the host nvcc would land on PATH
[ -n "$MODE" ] && [ -n "$IMAGE" ] || { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# The exact probe torch.compile makes, end to end (forced COLD so codegen runs). Base64 so the
# Python survives the bash -lc / singularity-exec / docker-run quoting layers intact.
read -r -d '' PYTEST <<'PY' || true
import torch
torch.compile(lambda x: (x.sin()+x.cos()).relu(), fullgraph=True)(
    torch.randn(512, 512, device="cuda")).sum().item()
print("COMPILE_OK")
PY
PYRUN="printf %s $(printf '%s' "$PYTEST" | base64 | tr -d '\n') | base64 -d | python3 -"

# run "<bash-snippet>"  [extra-bind-host-file]  — execute a snippet inside the container
run() {
  local snippet="$1" bindfile="${2:-}"
  if [ "$RUNNER" = docker ]; then
    local v=(); [ -n "$bindfile" ] && v=(-v "$bindfile:$VECTOR:ro")
    docker run --rm --gpus all "${v[@]}" -e TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
      --entrypoint bash "$IMAGE" -lc "$snippet"
  else
    local b=(); [ -n "$bindfile" ] && b=(--bind "$bindfile:$VECTOR")
    singularity exec --nv --containall "${b[@]}" --env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
      "$IMAGE" bash -lc "$snippet"
  fi
}

case "$MODE" in
  diagnose)
    echo "== nvcc resolution inside the container ($RUNNER: $IMAGE) =="
    run '
      echo "nvcc on PATH : $(command -v nvcc || echo NONE)"
      N=$(command -v nvcc 2>/dev/null); [ -n "$N" ] && N=$(readlink -f "$N")
      if [ -n "$N" ]; then
        echo "real path    : $N"
        ls -l "$N"; stat -c "mode=%A owner=%U:%G" "$N"; id
        echo "mount        :"; findmnt -T "$N" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
        echo "exec test    : $(nvcc --version >/dev/null 2>&1 && echo OK || echo "FAILS: $(nvcc --version 2>&1 | tail -1)")"
      fi
      echo "PATH         : $PATH"
    '
    ;;
  test)
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    printf '#!/bin/sh\necho "fake host nvcc"\n' > "$TMP/nvcc"
    chmod 0644 "$TMP/nvcc"        # readable, NOT executable == the noexec / no-+x host nvcc
    echo "== plant a NON-EXECUTABLE nvcc at $VECTOR and run a cold torch.compile =="
    OUT="$(run "$PYRUN" "$TMP/nvcc" 2>&1)"; RC=$?
    echo "$OUT" | grep -E "COMPILE_OK|Permission denied: 'nvcc'|InductorError|Errno 13" || echo "$OUT" | tail -5
    echo "----------------------------------------------------------------------"
    if echo "$OUT" | grep -q "COMPILE_OK"; then
      echo "RESULT: PASS — the container used its OWN nvcc; a non-executable host nvcc did NOT break it."
      exit 0
    elif echo "$OUT" | grep -qE "Permission denied: 'nvcc'|Errno 13"; then
      echo "RESULT: FAIL — VULNERABLE. torch.compile crashed on the non-executable nvcc."
      echo "        Fix: ship an executable nvcc first on PATH, or ensure the host nvcc is absent/executable."
      exit 1
    else
      echo "RESULT: INCONCLUSIVE (rc=$RC) — neither COMPILE_OK nor the nvcc error seen; inspect output above."
      exit 3
    fi
    ;;
  *)
    echo "unknown mode '$MODE' (use: diagnose | test)"; exit 2 ;;
esac
