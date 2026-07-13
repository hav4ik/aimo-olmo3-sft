#!/bin/bash
# NOTE: sanitized for publication. Set your serving endpoint + key before running:
#   export OLMO_ADDR=host:port   OLMO_API_KEY=your_key
# NOTE: temperature MUST be 1.0 (production) — this olmo checkpoint degenerates into repetitive loops
# below ~1.0. See memory olmo-temperature-1.0-required.
ADDR="${OLMO_ADDR:?set OLMO_ADDR=host:port}"
KEY="${OLMO_API_KEY:?set OLMO_API_KEY=your_api_key}"
# 1) wait (up to ~40 min) for the model to load
MODEL=""
for i in $(seq 1 160); do
  MODEL=$(curl -s -m 10 -H "Authorization: Bearer $KEY" http://$ADDR/v1/models 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MODEL" ] && break
  echo "$(date +%H:%M) waiting for model to load... (poll $i)"; sleep 15
done
if [ -z "$MODEL" ]; then echo "ABORT: model never came up after ~40min"; exit 1; fi
echo "MODEL UP: $MODEL"
# 2) ENSURE it is step17000 (user requirement) — abort if not
if ! echo "$MODEL" | grep -qi "step17000"; then
  echo "ABORT: served model id '$MODEL' does not contain 'step17000' — refusing to run"; exit 1
fi
echo "VERIFIED step17000: $MODEL"
# 3) AIME'25 no-tool then tool (temp 1.0, same prompts/settings as the others)
COMMON="--n_sessions 8 --server_addr $ADDR --api_key $KEY --model_name $MODEL --sandbox_port 6001 --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume"
echo "===== S17000 NOTOOL aime25 $(date +%H:%M) ====="
python3 run_eval.py --task boxed_notool --input bench_aime25.jsonl \
  --output_dir traces/bench_0207full_step17000_aime25_notool --max_parallel 48 $COMMON
echo "===== S17000 TOOL aime25 $(date +%H:%M) ====="
python3 run_eval.py --task boxed_tool --input bench_aime25.jsonl \
  --output_dir traces/bench_0207full_step17000_aime25_tool --max_parallel 24 --max_turns 64 \
  --python_timeout 60 --max_output_characters 600 $COMMON
echo "===== S17000 DONE $(date +%H:%M) ====="; touch traces/BENCH_0207FULL_S17000_DONE
