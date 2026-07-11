#!/bin/bash
# smolmo OLMo-core fast converter. Three modes (chosen by env), all download->convert->STOP (no upload):
#
#  SINGLE NODE (default): convert ALL 1024 shards in one process (stock shuffle, byte-identical to
#    stock olmo_thinker). Just: -e HF_TOKEN=... -v /out:/out
#
#  MULTI NODE: set PART_NUM (0..NUM_PARTS-1) and NUM_PARTS. Node k converts shards [k*1024/K,(k+1)*1024/K),
#    deterministically (no shuffle), writing node-prefixed parts. Run one container per node.
#    -e PART_NUM=0 -e NUM_PARTS=4 -e HF_TOKEN=... -v /out:/out
#
#  MERGE STATS: after all nodes uploaded their parts to one /out, set MERGE_STATS=1 -e NUM_PARTS=4.
#
# Common env: HF_TOKEN, PROCS (default nproc), WORK (default /out), OUTPUT, DATASET, DL_WORKERS.
set -euo pipefail

WORK="${WORK:-/out}"
OUTPUT="${OUTPUT:-$WORK/olmocore}"
DATASET="${DATASET:-chankhavu/smolmo-sft-v2-seqlen64k}"          # SOURCE repo (download + convert)
OUT_DATASET="${OUT_DATASET:-$DATASET}"                           # TARGET repo for upload (keep output separate)
PROCS="${PROCS:-$(nproc)}"
DL_WORKERS="${DL_WORKERS:-16}"
export HF_HOME="$WORK/.hf"; export HOME="$WORK/.hf"
export TOKENIZERS_PARALLELISM=false
export OPEN_INSTRUCT=/workspace/open-instruct
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$HF_HOME" "$WORK"
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ---- MERGE STATS mode ----
if [ "${MERGE_STATS:-0}" = "1" ]; then
  echo "[merge] combining stats_part_*.json in $OUTPUT (num_parts=${NUM_PARTS:?set NUM_PARTS})"
  python /opt/convert/convert_node.py --merge-stats --num-parts "$NUM_PARTS" --work "$WORK" --output "$OUTPUT"
  python /opt/convert/inspect_olmocore.py "$OUTPUT" || true
  echo "DONE merging. dataset_statistics.json written to $OUTPUT."
  exit 0
fi

echo "=================================================================="
echo " smolmo OLMo-core fast convert   dataset=$DATASET procs=$PROCS out=$OUTPUT"
df -h "$WORK" | tail -1

# ---- MULTI NODE mode ----
if [ -n "${PART_NUM:-}" ] && [ "${NUM_PARTS:-1}" -gt 1 ]; then
  echo " MULTI-NODE: part $PART_NUM of $NUM_PARTS (deterministic, split by parquet ids)"
  echo "=================================================================="
  SHARD_ARGS=()
  [ -n "${SHARD_START:-}" ] && SHARD_ARGS+=(--shard-start "$SHARD_START")   # explicit hard-coded range
  [ -n "${SHARD_END:-}" ]   && SHARD_ARGS+=(--shard-end "$SHARD_END")
  python /opt/convert/convert_node.py --part-num "$PART_NUM" --num-parts "$NUM_PARTS" \
    --dataset "$DATASET" --work "$WORK" --output "$OUTPUT" --procs "$PROCS" --dl-workers "$DL_WORKERS" \
    "${SHARD_ARGS[@]}"
  python /opt/convert/inspect_olmocore.py "$OUTPUT" || true
  cat <<EOF

==================================================================
 NODE $PART_NUM/$NUM_PARTS DONE — its parts are in: $OUTPUT
   token_ids_part_$(printf '%03d' "$PART_NUM")_*.npy  (+ labels_mask, csv.gz, stats_part_$(printf '%03d' "$PART_NUM").json)
 Inspect, then upload THIS node's parts to the olmocore/ folder of the OUTPUT repo ($OUT_DATASET):
   hf upload $OUT_DATASET $OUTPUT olmocore --repo-type dataset --include "*_$(printf '%03d' "$PART_NUM")_*" --include "stats_part_$(printf '%03d' "$PART_NUM").json"
 (part 0 also: --include "tokenizer/*")   [create it first if needed: hf repo create $OUT_DATASET --repo-type dataset]
 After ALL $NUM_PARTS nodes upload, run ONE merge pass:
   docker run --rm -e MERGE_STATS=1 -e NUM_PARTS=$NUM_PARTS -e HF_TOKEN=... -v /shared:/out smolmo-olmocore-fast:1.3
==================================================================
EOF
  exit 0
fi

# ---- SINGLE NODE mode (default) ----
echo " SINGLE-NODE: all shards, one process (byte-identical to stock olmo_thinker)"
echo "=================================================================="
DATA_GLOB="$WORK/ds/data/train-*.parquet"
if compgen -G "$DATA_GLOB" > /dev/null; then
  echo "[1/3] shards present at $WORK/ds — skipping download"
else
  echo "[1/3] downloading $DATASET in parallel (--max-workers $DL_WORKERS) ..."
  hf download "$DATASET" --repo-type dataset --local-dir "$WORK/ds" --max-workers "$DL_WORKERS"
fi
echo "[2/3] converting (procs=$PROCS) ..."
BEAKER_ASSIGNED_CPU_COUNT="$PROCS" python \
  "$OPEN_INSTRUCT/scripts/data/convert_sft_data_for_olmocore.py" \
  --dataset_mixer_list "$DATA_GLOB" 1.0 \
  --tokenizer_name_or_path allenai/Olmo-3-7B-Think --chat_template_name olmo_thinker \
  --output_dir "$OUTPUT" --dataset_local_cache_dir "$WORK/cache" \
  --max_seq_length 65536 --num_examples 0
echo "[3/3] conversion complete — inspecting (not uploading) ..."
python /opt/convert/inspect_olmocore.py "$OUTPUT" --show 1 || true
cat <<EOF

==================================================================
 DONE — output in: $OUTPUT
 INSPECT:  python /opt/convert/inspect_olmocore.py $OUTPUT --show 3
           cat $OUTPUT/dataset_statistics.json
 UPLOAD (when satisfied, WRITE token) to OUTPUT repo ($OUT_DATASET):
   hf upload $OUT_DATASET $OUTPUT olmocore --repo-type dataset
   [create it first if needed: hf repo create $OUT_DATASET --repo-type dataset]
==================================================================
EOF
