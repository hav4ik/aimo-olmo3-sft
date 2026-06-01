#!/bin/bash
# In-container Axolotl SFT run. Axolotl pulls the dataset + base model from the HF Hub
# (cached under HF_HOME=/data/training/hf_cache), so no offline data-prep is needed — just
# HF_TOKEN in the env. attn_implementation is auto-picked from GPU arch: sm_90 (H100/H200)
# -> flash_attention_3; anything else incl. sm_120 (RTX PRO 6000 / Blackwell) -> flex_attention
# (FA3 is Hopper-only). Override via ATTN_IMPL. Checkpoints under /data/training.
#   PRECISION=bf16|fp8  (default bf16) -> picks configs/olmo3-7b-<precision>.yaml
#   For a single-GPU smoke: SEQUENCE_LEN=2048 MAX_STEPS=10 (full recipe is seq 32768 / 2 ep).
set -euo pipefail
[ -f /workspace/axolotl-venv/bin/activate ] && source /workspace/axolotl-venv/bin/activate

DATA=/data/training
NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
PRECISION="${PRECISION:-bf16}"
CONFIG="${CONFIG:-/workspace/code/axolotl/configs/olmo3-7b-${PRECISION}.yaml}"

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
case "$CC" in 9.*) DEFATTN=flash_attention_3 ;; *) DEFATTN=flex_attention ;; esac
ATTN="${ATTN_IMPL:-$DEFATTN}"

# axolotl CLI overrides (key=value) layered on the yaml — keep paths under /data/training.
OVERRIDES=(
    "--attn_implementation=$ATTN"
    "--output_dir=$DATA/checkpoints/olmo3-7b-axolotl-$PRECISION"
    "--dataset_prepared_path=$DATA/last_run_prepared"
)
[ -n "${SEQUENCE_LEN:-}" ] && OVERRIDES+=("--sequence_len=$SEQUENCE_LEN")
[ -n "${MAX_STEPS:-}"    ] && OVERRIDES+=("--max_steps=$MAX_STEPS")
# FP8 on Blackwell sm_120: flex_attention can't pair with torchao fp8's compile path the
# same way FA does — if you hit issues, run the BF16 arm on RTX 6000 and FP8 on Hopper.

echo "[axolotl] $PRECISION | $NPROC GPU(s) cc=$CC | attn=$ATTN | config=$(basename "$CONFIG")"
exec accelerate launch --num_processes="$NPROC" -m axolotl.cli.train "$CONFIG" "${OVERRIDES[@]}"
