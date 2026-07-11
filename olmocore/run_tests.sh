#!/usr/bin/env bash
# Run ALL attention-sink / FA2 / FA3 / Ulysses-CP / ring / RoPE / RMSNorm unit tests in one shot.
#
# Two test sources:
#   1. olmo_core's own suite, baked into the image at <OLMo-core>/src/test (the real sink+CP kernels).
#   2. this recipe's tests under olmocore/tests (the fused-RMSNorm swap logic).
#
# Run it INSIDE the container (it has olmo_core + flash-attn + GPUs):
#
#   docker run --rm --gpus all --entrypoint bash \
#       chankhavu/olmo3-olmocore:cu128-fa2-sink \
#       /data/training/code/olmocore/run_tests.sh
#
# (or, if you're already in a running container, just `bash olmocore/run_tests.sh`).
#
# GPU notes:
#   * The sink/FA2/FA3 numeric tests need 1 CUDA GPU.
#   * The Ulysses-CP / ring / context-parallel tests self-skip unless >=2 GPUs are visible
#     (they spawn a small distributed group). Run on an 8-GPU box to exercise them.
#
# Flags:
#   --self-check   also run the standalone kernel consistency script (eager vs FA2/FA3, FA2 vs FA3),
#                  the same one OLMO_ATTN_SELFCHECK=1 runs as a pre-training pre-flight.
#   --quick        only the single-GPU sink/flash + recipe tests (skip the distributed suites).
#   -k EXPR, -x, … any extra args are forwarded to pytest (e.g. -k sink, -x, -v).
set -uo pipefail

SELF_CHECK=0
QUICK=0
PYTEST_ARGS=()
for a in "$@"; do
    case "$a" in
        --self-check) SELF_CHECK=1 ;;
        --quick)      QUICK=1 ;;
        *)            PYTEST_ARGS+=("$a") ;;
    esac
done

# --- locate the baked olmo_core test tree (import-based, so it survives path changes) ---------------
OLMO_CORE_ROOT="$(python -c 'import olmo_core,os;print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(olmo_core.__file__)))))' 2>/dev/null || true)"
TEST_ROOT="${OLMO_CORE_ROOT:+$OLMO_CORE_ROOT/src/test}"
if [ -z "${TEST_ROOT}" ] || [ ! -d "${TEST_ROOT}" ]; then
    echo "ERROR: could not locate olmo_core's test tree (import olmo_core failed or src/test missing)." >&2
    echo "       Are you running this inside the SFT container?" >&2
    exit 2
fi
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../olmocore

NGPU="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "=================================================================="
echo " olmo-core sink/FA/CP test runner"
echo "   olmo_core tests : ${TEST_ROOT}"
echo "   recipe tests    : ${RECIPE_DIR}/tests"
echo "   visible GPUs    : ${NGPU}"
[ "${NGPU}" -lt 1 ] && echo "   WARNING: 0 GPUs -> the FA2/FA3 sink numeric tests will skip/fail." >&2
[ "${NGPU}" -lt 2 ] && echo "   NOTE: <2 GPUs -> Ulysses/ring/context-parallel tests will self-skip." >&2
echo "=================================================================="

# --- the suites, in dependency order (single-GPU first, distributed last) --------------------------
# Paths are relative to $TEST_ROOT; missing ones are warned + skipped (fork drift tolerance).
SINGLE_GPU=(
    "nn/attention/attention_sink_test.py"          # sink module: read/train/export, per-head logit
    "nn/attention/attention_sink_flash_test.py"    # FA2 in-kernel vs FA3 vs eager post-correction
    "nn/attention/attention_test.py"               # attention block wiring
    "nn/rope_test.py"                               # RoPE (incl. intra-doc for CP)
    "nn/layer_norm_test.py"                         # RMSNorm parity incl. D=5120 (fused-rms swap target)
)
DISTRIBUTED=(
    "nn/attention/attention_sink_ulysses_test.py"  # sink correctness under Ulysses CP all-to-all
    "nn/attention/ring_test.py"                     # ring-CP attention
    "distributed/parallel/context_parallel_test.py" # CP mesh / sharding
)

declare -a SUITES=()
for t in "${SINGLE_GPU[@]}"; do SUITES+=("$TEST_ROOT/$t"); done
if [ "${QUICK}" -eq 0 ]; then
    for t in "${DISTRIBUTED[@]}"; do SUITES+=("$TEST_ROOT/$t"); done
fi

# keep only paths that exist (tolerate fork drift), warn on the rest
declare -a EXIST=()
for s in "${SUITES[@]}"; do
    if [ -f "$s" ]; then EXIST+=("$s"); else echo "  (skip, not found: ${s#$TEST_ROOT/})"; fi
done

RC=0

# pytest exit codes: 0=pass, 1=failures, 5=no tests collected. Treat 5 as non-fatal (a -k filter that
# matched nothing, or all-deselected) — only a real failure (1/2/3/4) fails the runner.
_grade() {  # $1 = pytest rc
    if [ "$1" -eq 0 ]; then return 0; fi
    if [ "$1" -eq 5 ]; then echo "  (no tests matched/collected — not counted as a failure)"; return 0; fi
    RC=1
}

echo
echo ">>> [1/3] olmo_core sink/FA/CP suites"
if [ "${#EXIST[@]}" -gt 0 ]; then
    ( cd "$OLMO_CORE_ROOT" && python -m pytest "${EXIST[@]}" -q -rs "${PYTEST_ARGS[@]}" ); _grade $?
else
    echo "  no olmo_core test files found — check the image." ; RC=1
fi

echo
echo ">>> [2/3] recipe tests (fused-RMSNorm swap)"
if [ -d "${RECIPE_DIR}/tests" ]; then
    ( cd "$RECIPE_DIR/.." && python -m pytest "olmocore/tests" -q -rs "${PYTEST_ARGS[@]}" ); _grade $?
else
    echo "  ${RECIPE_DIR}/tests not found — skipping."
fi

echo
echo ">>> [3/3] kernel self-check (eager vs FA2/FA3, FA2 vs FA3)"
if [ "${SELF_CHECK}" -eq 1 ]; then
    python "$TEST_ROOT/nn/attention/attention_sink_flash_test.py" || RC=1
else
    echo "  skipped (pass --self-check to run the standalone consistency check)."
fi

echo
echo "=================================================================="
if [ "$RC" -eq 0 ]; then echo " RESULT: all suites passed (skips are expected on <2 GPUs / missing backends)."
else echo " RESULT: FAILURES above — scroll up for the failing suite."; fi
echo "=================================================================="
exit "$RC"
