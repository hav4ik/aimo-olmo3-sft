#!/bin/bash
# NOTE: sanitized for publication. Set your serving endpoint + key before running:
#   export OLMO_ADDR=host:port   OLMO_API_KEY=your_key
# NOTE: temperature MUST be 1.0 (production). This olmo checkpoint degenerates into repetitive
# loops below ~1.0 (0.6 truncates & tanks accuracy). Do NOT lower it. See memory olmo-temperature-1.0-required.
ADDR="${OLMO_ADDR:?set OLMO_ADDR=host:port}"
KEY="${OLMO_API_KEY:?set OLMO_API_KEY=your_api_key}"
MODEL=chankhavu/olmo_32b_fp8_niiattempt2_step20500_20260620144800
COMMON="--n_sessions 8 --server_addr $ADDR --api_key $KEY --model_name $MODEL --sandbox_port 6001 --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume"
echo "===== S20500 NOTOOL aime25 $(date +%H:%M) ====="
python3 run_eval.py --task boxed_notool --input bench_aime25.jsonl \
  --output_dir traces/bench_0207full_step20500_aime25_notool --max_parallel 48 $COMMON
echo "===== S20500 TOOL aime25 $(date +%H:%M) ====="
python3 run_eval.py --task boxed_tool --input bench_aime25.jsonl \
  --output_dir traces/bench_0207full_step20500_aime25_tool --max_parallel 24 --max_turns 64 \
  --python_timeout 60 --max_output_characters 600 $COMMON
echo "===== S20500 DONE $(date +%H:%M) ====="; touch traces/BENCH_0207FULL_S20500_DONE
