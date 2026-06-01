#!/bin/bash
# OLMo-core 7B SFT — FP8 (precision experiment).
#
# Thin wrapper over launch_7b_hopper.sh: torchao float8 ROWWISE on the feed-forward
# linears, KEEPING ALL attention + lm_head + embeddings in high precision (DeepSeek-V3
# recipe; see fp8_attention_ignores in Olmo-3-7B-SFT-local.py). FA3 + auto-CP stay on, so
# this is IDENTICAL to launch_7b_bf16.sh except OLMO_FP8 — the A/B isolates precision.
#   OLMO_FP8=rowwise (default)           - per-channel scaling: the finer-grained,
#                                          accuracy-preserving recipe. DeepSeek-V3 (<0.25%
#                                          loss gap) and Unsloth both find fine-grained
#                                          (per-channel/block) FP8 stable where per-tensor
#                                          is risky — so we test viability with this first.
#   OLMO_FP8=tensorwise                  - per-tensor: fastest (+ FSDP fp8 all-gather) but
#                                          coarsest/riskiest; try only after rowwise is clean.
# Validate the BF16 arm first, then compare loss curves / grad-norm / spikes.
set -euo pipefail
export OLMO_FP8="${OLMO_FP8:-rowwise}"
export RUN_NAME="${RUN_NAME:-olmo3-7b-sft-fp8}"
exec "$(dirname "$0")/launch_7b_hopper.sh" "$@"
