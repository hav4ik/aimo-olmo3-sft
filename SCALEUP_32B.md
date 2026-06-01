# 32B scale-up plan (do AFTER the 7B is validated)

The 32B is ~256-GPU-scale and expensive — **derisk on the 7B first.** Don't start the 32B until:
1. Both frameworks train the **real dataset** at 7B without divergence (loss trends down, no spikes;
   for OLMo-core watch the SkipStepAdamW skip count).
2. If using FP8 at 7B: its loss curve tracks BF16 (see `STABILITY.md`).
3. A 7B checkpoint passes a quick eval / smell-test (the continue-tune didn't wreck the model).

## The 32B recipe ≠ the 7B recipe (AI2 `32b_think_sft.sh`)
| knob | 7B (`7b_think_sft`) | **32B (`32b_think_sft`)** |
|---|---|---|
| lr | 5e-5 | **1e-4** |
| global batch | 1,048,576 tok (32 seqs) | **4,194,304 tok (128 seqs)** |
| epochs | 2 | 2 |
| seq len | 32768 | 32768 |
| AI2 shape | 8 nodes × 8 = 64 GPU | **32 nodes × 8 = 256 GPU** |
Optimizer (SkipStepAdamW), hsdp, selective AC, compile, bf16 — same as the 7B. The higher lr +
4× batch are the real differences; our `olmo3-32b-{bf16,fp8}.yaml` already encode them.

## What's already ready (no work needed)
- **Model:** `allenai/Olmo-3.1-32B-Think` (continue-tune, like the 7B). Same dolma2 tokenizer +
  same 3-sliding:1-full attention pattern (`[4096,4096,4096,-1]`, force-full last) as the 7B.
- **Axolotl 32B:** `axolotl/configs/olmo3-32b-{bf16,fp8}.yaml` — recipe-accurate (lr 1e-4, 4.19M
  batch, seq 32768, `tokenizer_default` think template, parquet input). Run with `MODEL_SIZE=32b`.
- **Data prep:** the SAME `data_prep/prepare.sh` — the dolma2 tokenizer is shared across Olmo 3
  sizes, so the same prepped `messages.parquet` / `.npy` work for 7B and 32B. Just reuse `DATASET_NAME`.
- **Deploy image:** the same `hav4ik/olmo3-axolotl:cu130` (axolotl 32B works today via `MODEL_SIZE=32b`).

## TODO: wire the OLMo-core 32B reference path (the held work)
Axolotl 32B runs now; the **reference** engine (OLMo-core) 32B is NOT wired yet — the trainer
`olmocore/sft_scripts/Olmo-3-7B-SFT-local.py` is 7B-only (`olmo3_7B` factory). To add it:
1. **`olmocore/sft_scripts/Olmo-3-32B-SFT-local.py`** — beaker-stub AI2's upstream
   `OLMo-core/src/scripts/train/sft/Olmo-3-32B-SFT.py` (it EXISTS) using the SAME ~4 minimal diffs
   as the 7B copy: beaker stubs, `GPUS_PER_NODE` from `LOCAL_WORLD_SIZE`, opt-in
   `OLMO_ATTN_BACKEND`/`OLMO_MODEL_DTYPE`/`OLMO_FP8`. The FP8 exclusion `fp8_attention_ignores`
   is arch-agnostic (it walks a meta build), so it auto-adapts to the 32B's full-attention layers —
   no hardcoded indices to change. Keep "a no-env run == AI2 bit-for-bit."
2. **`olmocore/run.sh`** — add a `MODEL_SIZE`/arch switch: pick the 32B script + `--model-arch
   olmo3_32b` for `STAGE=convert`, and the 32B recipe defaults (lr **1e-4**, `global_batch_size`
   **4194304**). (Today it hardcodes the 7B script + `olmo3_7b` + lr 5e-5 / 1048576.)
3. **`data_prep/convert_hf_to_olmocore.sh`** — `MODEL_ARCH=olmo3_32b`, `HF_MODEL=Olmo-3.1-32B-Think`,
   output e.g. `/data/training/checkpoints/olmocore-olmo3-32b-think`.
4. Verify on a meta build that `olmo3_32B`'s full-attention FQNs come out as expected (FP8 exclusion).

## Hardware / memory sizing
- 32B full-FT fp32-AdamW state ≈ 32B × 16 B ≈ **512 GB** (master + grad + m + v) — FSDP shards it
  across ranks. It *fits* on ~8×H200 (1.1 TB aggregate) with activation room, but the 4.19M-tok
  batch + throughput push you to **multi-node** (AI2 used 256 GPUs). Practical minimum: 16–32×H200
  with FSDP; more for speed. Use `NNODES`/`NODE_RANK`/`HEAD_NODE_IP` (OLMo-core run.sh) for multi-node.
- Context parallelism auto-engages at seq 32768 (`cp_degree≥2`) on the OLMo-core side.
- The axolotl 32B accum targets 128 seqs (32 GPUs → accum 4; 64 → 2; 16 → 8).

## Running the 32B (once wired)
```bash
# 1. prep the real data (same as 7B; one parquet works for both sizes)
./data_prep/prepare.sh --name mymath --input <real-dataset> --template olmo_thinker --max-seq 32768
# 2. convert the 32B checkpoint once (OLMo-core)
docker run --rm --gpus all -v /data/training:/data/training -e HF_TOKEN=$HF_TOKEN \
  -e FRAMEWORK=olmocore -e MODEL_SIZE=32b -e STAGE=convert hav4ik/olmo3-olmocore:cu130
# 3a. axolotl 32B (works today) — multi-node example
docker run ... -e FRAMEWORK=axolotl -e MODEL_SIZE=32b -e PRECISION=bf16 -e DATASET_NAME=mymath \
  -e NNODES=2 -e NODE_RANK=0 -e HEAD_NODE_IP=... hav4ik/olmo3-axolotl:cu130
# 3b. OLMo-core 32B — after the wiring above
docker run ... -e FRAMEWORK=olmocore -e MODEL_SIZE=32b -e PRECISION=bf16 -e DATASET_NAME=mymath ...
```
Always do a short `MAX_STEPS=20` pipeline smoke before the full 2-epoch run — at 256 GPUs a failed
launch is expensive.

## Risks
- **Continue-tune at 32B is costly to redo** — confirm the 7B continue-tune behavior (does SFT on
  the math data improve, not degrade, the Think model?) before committing 32B compute.
- **FP8 at 32B:** runs on H200 (Hopper) → the proven `flash_3` path (not the untested Blackwell+flex
  combo). Still validate FP8-vs-BF16 loss parity on a short 32B run.
- **lr 1e-4 is aggressive** on a continue-tune; watch early loss/grad-norm closely.
