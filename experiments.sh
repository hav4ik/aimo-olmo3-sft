#!/bin/bash
# Experiment registry. ONE knob — EXPERIMENT=<size>_<precision>_<variant> (e.g. 7b_bf16_cot,
# 32b_fp8_cot) — resolves to the full run config, so the launch command is just the topology env
# (MASTER_ADDR/PORT/WORLD_SIZE/GLOBAL_RANK, set by the cluster) + `-e EXPERIMENT=...`. Sourced by
# entrypoint.sh. Any var already set in the environment is preserved (explicit override wins).
# Add experiments by extending the cases below. The tokenized dataset is hosted on HF and pulled
# at runtime (same dolma2 tokenization for 7B/32B, so a variant maps to ONE dataset repo).
resolve_experiment() {
    local exp="${1:?EXPERIMENT is empty}" size prec variant ds
    IFS=_ read -r size prec variant <<< "$exp"
    [ -n "$size" ] && [ -n "$prec" ] && [ -n "$variant" ] || {
        echo "[experiments] bad EXPERIMENT='$exp' (want <size>_<precision>_<variant>, e.g. 32b_fp8_cot)"; return 2; }

    case "$size" in 7b|32b) : ;; *) echo "[experiments] unknown size '$size' (want 7b|32b)"; return 2 ;; esac
    case "$prec" in bf16|fp8) : ;; *) echo "[experiments] unknown precision '$prec' (want bf16|fp8)"; return 2 ;; esac
    # variant -> the PRE-TOKENIZED olmo-core dataset on HF: repo (ds) + subfolder holding the
    # token_ids_part_*.npy / labels_mask_*.npy (sub). Add a case line per dataset you host.
    local sub
    case "$variant" in
        cot)  ds="chankhavu/smolmo-sft-olmocore-pretokenized"; sub="olmocore" ;;
        *) echo "[experiments] unknown variant '$variant' — add it to experiments.sh"; return 2 ;;
    esac

    export MODEL_SIZE="${MODEL_SIZE:-$size}"
    export PRECISION="${PRECISION:-$prec}"
    export DATASET_NAME="${DATASET_NAME:-$variant}"
    export DATASET_HF="${DATASET_HF:-$ds}"
    export DATASET_SUBDIR="${DATASET_SUBDIR:-$sub}"
    export RUN_NAME="${RUN_NAME:-olmo3-${size}-${prec}-${variant}}"
}
