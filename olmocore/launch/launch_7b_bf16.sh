#!/bin/bash
# OLMo-core 7B SFT — BF16 (precision baseline).
#
# Thin wrapper over launch_7b_hopper.sh with FP8 OFF => bf16, exactly AI2's (and
# Nemotron Nano's) SFT precision. FA3 packed-varlen + automatic CP stay on, so this
# is the BF16 arm of the BF16-vs-FP8 stability A/B (the FP8 arm = launch_7b_fp8.sh
# differs ONLY in OLMO_FP8). All shape env still applies: NPROC_PER_NODE, NNODES,
# NODE_RANK, HEAD_NODE_IP (e.g. NPROC_PER_NODE=8 for 8xH200).
set -euo pipefail
export OLMO_FP8=""                                  # off => bf16
export RUN_NAME="${RUN_NAME:-olmo3-7b-sft-bf16}"
exec "$(dirname "$0")/launch_7b_hopper.sh" "$@"
