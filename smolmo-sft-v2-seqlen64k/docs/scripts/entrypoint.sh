#!/bin/bash
# Self-contained entrypoint: downloads the HF dataset and runs the faithful olmo_thinker
# olmocore conversion. Configure via env (HF_TOKEN, PROCS, OUTPUT, DATASET) or pass extra
# flags (e.g. --num-shards K --shard-index k, or --combine) forwarded to the converter.
# NOTE: mount a large writable dir to /out (>= ~400GB for the full dataset): it holds the HF
# download cache, the tokenized cache, and the output memmaps.
set -euo pipefail
OUTPUT="${OUTPUT:-/out/olmocore}"
export HF_HOME="/out/.hf"; export HOME="/out/.hf"     # FORCE writable (base image defaults HOME/HF_HOME to /data)
export TOKENIZERS_PARALLELISM=false; export OPEN_INSTRUCT=/workspace/open-instruct
mkdir -p "$HF_HOME"
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PROCS="${PROCS:-$(nproc)}"; DATASET="${DATASET:-chankhavu/smolmo-sft-v2-seqlen64k}"
echo "[entrypoint] dataset=$DATASET procs=$PROCS output=$OUTPUT cores=$(nproc) extra: $*"
exec python /opt/convert/convert_hf_to_olmocore.py --mode python \
    --dataset "$DATASET" --output "$OUTPUT" --procs "$PROCS" "$@"
