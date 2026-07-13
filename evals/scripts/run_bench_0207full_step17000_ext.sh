#!/bin/bash
# NOTE: sanitized for publication. Set your serving endpoint + key before running:
#   export OLMO_ADDR=host:port   OLMO_API_KEY=your_key
# temperature MUST be 1.0 (production; low-temp degenerates). max_tokens 0 = full context.
ADDR="${OLMO_ADDR:?set OLMO_ADDR=host:port}"
KEY="${OLMO_API_KEY:?set OLMO_API_KEY=your_api_key}"
MODEL=chankhavu/smolmo-32b-nvfp4-step17000
echo "$(date +%H:%M) ext: waiting for step17000 aime25 to finish (BENCH_0207FULL_S17000_DONE)..."
while [ ! -f traces/BENCH_0207FULL_S17000_DONE ]; do sleep 60; done
# only proceed if aime25 went well (tool run produced a full set)
if [ "$(ls traces/bench_0207full_step17000_aime25_tool/*.jsonl 2>/dev/null|wc -l)" -lt 28 ]; then
  echo "ABORT ext: aime25 tool incomplete (<28/30) — not proceeding to aime26/hmmt"; exit 1; fi
echo "$(date +%H:%M) ext: aime25 OK, starting aime26 + hmmt"
COMMON="--n_sessions 8 --server_addr $ADDR --api_key $KEY --model_name $MODEL --sandbox_port 6001 --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume"
for bench in aime26 hmmt_feb2025; do
  echo "===== S17000 NOTOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_notool --input bench_${bench}.jsonl --output_dir traces/bench_0207full_step17000_${bench}_notool --max_parallel 48 $COMMON
done
for bench in aime26 hmmt_feb2025; do
  echo "===== S17000 TOOL $bench $(date +%H:%M) ====="
  python3 run_eval.py --task boxed_tool --input bench_${bench}.jsonl --output_dir traces/bench_0207full_step17000_${bench}_tool --max_parallel 24 --max_turns 64 --python_timeout 60 --max_output_characters 600 $COMMON
done
echo "===== S17000 EXT DONE $(date +%H:%M) ====="; touch traces/BENCH_0207FULL_S17000_EXT_DONE
