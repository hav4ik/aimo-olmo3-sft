# OLMo-core SFT — RUNBOOK (Olmo 3)

Environment-only Docker image + bind-mounted runs dir for full-parameter SFT of
**Olmo 3** on `allenai/tulu-3-sft-personas-math`. Mirrors the axolotl track
(`../olmo3-axolotl-sft`) so the two frameworks can be compared apples-to-apples.

## Models (per project decision)

| Tier | Model | Notes |
|---|---|---|
| dev / mid / prod-7b | `allenai/Olmo-3-7B-Think` | 7B, `Olmo3ForCausalLM`, dolma2 tok (vocab 100278) |
| prod-32b | `allenai/Olmo-3.1-32B-Think` | 32B thinking model |

OLMo-core loads its **own** distributed-checkpoint format, not HF models, so the
HF weights are converted first (Step 2).

## Image (AI2's official Dockerfile, cu130)

The image targets **Hopper (H100/H200)** — that is the real training hardware; the
2×3090 box is only for smoke tests. So we build **AI2's own `src/Dockerfile`
UNCHANGED** (their canonical recipe: cu130 torch 2.10 + FA2 + FA3 + TransformerEngine
+ grouped-gemm + liger + cloud SDKs + MLNX OFED). Two-step build:

```bash
cd ../OLMo-core
# 1) AI2's official deps image, unchanged, for cu130:
docker build -f src/Dockerfile \
  --build-arg CUDA_VERSION=13.0.2 --build-arg CUDA_VERSION_PATH=cu130 \
  --build-arg TORCH_VERSION=2.10.0 -t olmo-core:cu130-2.10.0 .
# 2) Bake our olmo_core source on top (the official image is deps-only by design):
docker build -f docker/Dockerfile.runtime \
  --build-arg BASE_IMAGE=olmo-core:cu130-2.10.0 -t olmo-core-sft:cu130 .
```

The cu130 image runs on the 3090 dev box too (its driver supports cu130), via the
`torch` SDPA backend — flash is Hopper-only. Transfer to the cluster with
`docker save olmo-core-sft:cu130 | ssh node 'docker load'`.

Validation:
```bash
docker run --rm --gpus all olmo-core-sft:cu130 python -c "import olmo_core, torch; \
from olmo_core.nn.attention import AttentionBackendName as A; \
print('torch', torch.__version__, '| backends', [b.name for b in A])"
```

## Pinned versions (observed)

- OLMo-core `2.5.0` (fork `hav4ik/OLMo-core`, branch `main`)
- torch `2.8.0+cu128` · transformers `5.9.0` · datasets `4.8.5` · liger-kernel `0.8.0`
- dolma2 tokenizer: vocab 100278, padded 100352; `<|im_start|>`=100264,
  `<|im_end|>`=100265, `<|endoftext|>` (eos) = 100257

## Workflow

### 1. Data prep — OLMo's OWN canonical tool (open-instruct)

We do NOT hand-roll tokenization. Olmo 3 was post-trained with open-instruct, so
SFT data is tokenized by its `convert_sft_data_for_olmocore.py` + registered
`olmo` chat template, in a dedicated data-prep image (no vllm/flash-attn):

```bash
# build the data-prep image once (../../open-instruct/Dockerfile.dataprep)
cd ../open-instruct && docker build -f Dockerfile.dataprep -t open-instruct-dataprep:0.1.0 .

# tokenize -> OLMo-core .npy
cd ../olmo3-olmocore-runs && ./data_prep/convert_sft_data.sh
```

This gives the canonical masking (assistant content + `<|endoftext|>` eos trained;
system/user **and the assistant header** masked) and OLMo-core `.npy` format.
Output `/data/datasets/tulu-math-olmocore-oi/`: `token_ids_part_*.npy`,
`labels_mask_part_*.npy`, stats. Observed: 2000 ex, 2.66 M tokens, **73.3%
trainable**, eos `<|endoftext|>` (100257) marks doc boundaries. The axolotl side
uses the SAME olmo template via its native machinery — see `../DATA.md` for the
token-level audit and cross-framework masking agreement.

### 2. Convert HF → OLMo-core checkpoint

```bash
HF_MODEL=allenai/Olmo-3-7B-Think ./data_prep/convert_hf_to_olmocore.sh
```

The converter `cached_path`s `<input>/config.json`, so it needs a **local dir**
(an HF cache snapshot dir works), not a bare repo id. Validation is skipped
(it loads two 7B models — too much for a 62 GB box); we validate via training-loss
parity instead. Output: `/data/checkpoints/olmocore-olmo3-7b-think/model_and_optim/`
(28 GB fp32 distcp). The trainer auto-resolves the `model_and_optim` subdir.

### 3. Dev smoke (2× RTX 3090)

```bash
OLMO_OPTIM=noop OLMO_OFFLOAD=0 OLMO_MODEL_DTYPE=bfloat16 \
  SEQ_LEN=512 MAX_STEPS=3 ./launch/launch_7b_dev.sh
```

(Equivalently the `docker run … olmo3_sft_local.py … --seq_len 512 --max_steps 3`
form.) Observed: model builds (7.299 B), FSDP dp=2, checkpoint loads, forward +
backward run, **CE loss ≈ 0.71** (PPL 2.0), peak ≈ 21 GiB/GPU, checkpoint saves,
clean exit.

## ⚠️ Dev-tier limitation (2× RTX 3090): no real optimizer step

Full 7B AdamW SFT does **not** fit 2× 24 GB: FSDP-sharded fp32 AdamW state alone
is ~29 GB/GPU. The usual fix is FSDP2 **CPU offload**, but **OLMo-core 2.5.0 does
not support it** — it has no offload config field, and injecting `CPUOffloadPolicy`
into `fully_shard` cascades into failures (weights materialize on GPU not CPU;
the model then reports CPU as its device so inputs land on CPU while params
unshard to GPU; `index_select` device mismatch). This matches the original task's
"Edit G" caveat that offload "may require a code path change."

So the dev tier runs with **`OLMO_OPTIM=noop`** (zero-state optimizer): a real
forward + backward executes (the step-1 loss is the true model loss, for parity),
but **weights are not updated**. This validates the whole pipeline — image,
conversion, tokenization/masking, dataset packing, model build, FSDP wrap,
checkpoint load/save, forward/backward — *except* the weight update.

**Real loss-decreasing 7B training runs on the mid tier (4× H100):** 320 GB fits
fp32 AdamW without offload. There, set `OLMO_OPTIM=adamw OLMO_MODEL_DTYPE=float32
OLMO_OFFLOAD=0`, bump `SEQ_LEN`, and (once flash-attn-3 is in the image) the
flash backend restores true per-document attention isolation.

## Cross-framework parity (same model, same data, step-1 CE loss)

Both on the canonical olmo recipe (same template, masking, eos), seq 2048:

| Framework | step-1/2/3 CE | optimizer | attention | weight update |
|---|---|---|---|---|
| axolotl (`../olmo3-axolotl-sft`) | **0.305 / 0.409 / 0.357** | AdamW (fp32, CPU offload) | flex (isolated) | yes |
| OLMo-core (this) | **0.154 / 0.246 / 0.267** | NoOp | torch SDPA (causal) | no |

Both ~0.15–0.40 on the same RL-Zero-**Math** model (already strong on math, hence
the low loss) vs ~10 for an untrained model — similar initial results. Residual
gap = attention (axolotl isolates packed samples; OLMo-core's torch backend uses
causal cross-document attention), batch composition, and real-update vs NoOp. A
tighter comparison needs the flash backend + real optimizer (Hopper tier).

## Files

- `sft_scripts/olmo3_sft_local.py` — beaker-free local trainer (the upstream SFT
  script hard-imports `beaker`). Env knobs: `OLMO_OFFLOAD`, `OLMO_OPTIM`,
  `OLMO_MODEL_DTYPE`, `OLMO_SFT_SAVE_ROOT`, `WANDB_*`.
- `data_prep/prepare_tulu_math.py` — tokenized .npy with axolotl-parity masking.
- `data_prep/convert_hf_to_olmocore.sh` — HF → OLMo-core checkpoint.
- `launch/launch_7b_dev.sh` — dev-tier launcher.
- `.env` (gitignored) / `.env.template` — secrets (none needed for the smoke).
