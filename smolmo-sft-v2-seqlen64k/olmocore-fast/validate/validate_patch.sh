#!/usr/bin/env bash
# Reproducible byte-identical proof: run the STOCK (unpatched) open-instruct converter and the
# PATCHED converter on the same inputs, then diff the .npy outputs. Exit non-zero on ANY diff.
#
# Usage: validate_patch.sh <sample.parquet | "f1.parquet,f2.parquet,...">  <work_dir>
# Requires: docker image open-instruct-dataprep:0.1.0, the patched dt.combined.py.
set -euo pipefail
INPUT="${1:?input parquet (single file, glob, or comma-list)}"
WORK="${2:-/tmp/tok_validate}"
IMG=open-instruct-dataprep:0.1.0
PATCHED="${PATCHED_DT:?path to patched dataset_transformation.py}"
HF_TOK=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
DATA_HOST="${DATA_HOST:-/mnt/data/proof-redesign/smolmo-math-cot-sft/data}"

run () {  # $1=tag  $2=extra docker mounts
  docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/work/.hf -e HF_HOME=/work/.hf -e TOKENIZERS_PARALLELISM=false -e BEAKER_ASSIGNED_CPU_COUNT=16 \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOK" \
    -v "$WORK":/work -v "$DATA_HOST":/data:ro $2 \
    "$IMG" python /workspace/open-instruct/scripts/data/convert_sft_data_for_olmocore.py \
    --dataset_mixer_list "$INPUT" 1.0 \
    --tokenizer_name_or_path allenai/Olmo-3-7B-Think --chat_template_name olmo_thinker \
    --output_dir "/work/out_$1" --dataset_local_cache_dir "/work/cache_$1" \
    --max_seq_length 65536 --num_examples 0 >"$WORK/log_$1.txt" 2>&1
}

echo "== STOCK (unpatched) =="; run stock ""
echo "== PATCHED =="; run patched "-v $PATCHED:/workspace/open-instruct/open_instruct/dataset_transformation.py:ro"

python3 - "$WORK" <<'PY'
import sys, glob, numpy as np, hashlib, os
work=sys.argv[1]
def cat(tag,name,dt):
    fs=sorted(glob.glob(f"{work}/out_{tag}/{name}_part_*.npy"))
    return np.concatenate([np.fromfile(f,dtype=dt) for f in fs]) if fs else np.array([],dtype=dt)
ok=True
for name,dt in [("token_ids",np.uint32),("labels_mask",np.bool_)]:
    a=cat("stock",name,dt); b=cat("patched",name,dt)
    same = a.shape==b.shape and bool(np.array_equal(a,b))
    ha=hashlib.sha256(a.tobytes()).hexdigest()[:16]; hb=hashlib.sha256(b.tobytes()).hexdigest()[:16]
    print(f"{name:12s} stock={a.shape} patched={b.shape} sha={ha}/{hb} IDENTICAL={same}")
    ok = ok and same
print("RESULT:", "PASS — byte identical" if ok else "FAIL — DIFFERENCES")
sys.exit(0 if ok else 1)
PY
