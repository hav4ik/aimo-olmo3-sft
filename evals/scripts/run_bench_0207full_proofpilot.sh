#!/bin/bash
# NOTE: sanitized for publication. Set your serving endpoint + key before running:
#   export OLMO_ADDR=host:port   OLMO_API_KEY=your_key
# NOTE: temperature MUST be 1.0 (production). This olmo checkpoint degenerates into repetitive
# loops below ~1.0 (0.6 truncates & tanks accuracy). Do NOT lower it. See memory olmo-temperature-1.0-required.
ADDR="${OLMO_ADDR:?set OLMO_ADDR=host:port}"
KEY="${OLMO_API_KEY:?set OLMO_API_KEY=your_api_key}"
MODEL=chankhavu/smolmo-32b-sft-merged-proofpilot-nvfp4
COMMON="--n_sessions 8 --server_addr $ADDR --api_key $KEY --model_name $MODEL --sandbox_port 6001 --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume"
for bench in aime25 aime26 hmmt_feb2025; do
  echo "===== PP NOTOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_notool --input bench_${bench}.jsonl \
    --output_dir traces/bench_0207full_proofpilot_${bench}_notool --max_parallel 48 $COMMON
done
for bench in aime25 aime26 hmmt_feb2025; do
  echo "===== PP TOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_tool --input bench_${bench}.jsonl \
    --output_dir traces/bench_0207full_proofpilot_${bench}_tool --max_parallel 24 --max_turns 64 \
    --python_timeout 60 --max_output_characters 600 $COMMON
done
echo "===== PP ALL DONE $(date +%H:%M) ====="; touch traces/BENCH_0207FULL_PP_DONE
