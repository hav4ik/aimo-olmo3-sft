#!/bin/bash
# Minimal, parameterized benchmark runner for AIME'25 / AIME'26 / HMMT-Feb'25.
# No secrets: point it at your own OpenAI-compatible endpoint via env vars.
#
#   export OLMO_ADDR=host:port          # your vLLM / SGLang server (OpenAI-compatible)
#   export OLMO_API_KEY=your_api_key
#   export OLMO_MODEL=chankhavu/smolmo-32b-nvfp4-step17000
#   ./run_bench.sh                      # runs all 3 benchmarks, no-tool then tool
#
# Tool mode additionally needs a NeMo-Skills sandbox reachable on $SANDBOX_PORT
# (see README "Tool mode"). Verify it first with:  python3 check_sandbox.py 127.0.0.1:6001
set -euo pipefail

ADDR="${OLMO_ADDR:?set OLMO_ADDR=host:port}"
KEY="${OLMO_API_KEY:?set OLMO_API_KEY=your_api_key}"
MODEL="${OLMO_MODEL:?set OLMO_MODEL=<hf model id served by the endpoint>}"
SANDBOX_PORT="${SANDBOX_PORT:-6001}"
OUT="${OUT_DIR:-traces}"
BENCHES="${BENCHES:-aime25 aime26 hmmt_feb2025}"

# ⚠️ temperature MUST be 1.0 for these Olmo-3 32B checkpoints. They degenerate into
# repetitive loops below ~1.0 (0.6 truncates and tanks accuracy). Do NOT lower it.
# --max_tokens 0 fills the full 65k context (no artificial output cap).
COMMON="--n_sessions ${N_SESSIONS:-8} --server_addr $ADDR --api_key $KEY \
  --model_name $MODEL --sandbox_port $SANDBOX_PORT \
  --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume"

for bench in $BENCHES; do
  echo "===== NO-TOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_notool --input "bench_${bench}.jsonl" \
    --output_dir "$OUT/${bench}_notool" --max_parallel 48 $COMMON
done

for bench in $BENCHES; do
  echo "===== TOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_tool --input "bench_${bench}.jsonl" \
    --output_dir "$OUT/${bench}_tool" --max_parallel 24 --max_turns 64 \
    --python_timeout 60 --max_output_characters 600 $COMMON
done

echo "===== ALL DONE $(date +%H:%M) — grade with: python3 grade_bench.py $OUT/aime25_notool ====="
