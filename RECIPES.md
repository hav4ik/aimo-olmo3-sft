# SFT recipe comparison — Axolotl vs OLMo-core

Recipe hyperparameters follow AI2's reference (`open-instruct/scripts/train/olmo3/7b_think_sft.sh`).
**Chat format: the canonical `olmo` template** (both frameworks). Olmo 3 declares
**two eos tokens** — `<|im_end|>` (100265) and `<|endoftext|>` (100257) — and
**stops on either** (verified in OLMo-core `nn/hf/checkpoint.py` and open-instruct
`model_utils.get_olmo3_generation_config`). So the served model is ChatML-inference
compatible **without changing the data format**: it ends turns with `<|endoftext|>`,
which is in the stop set, so any stack respecting `generation_config` stops correctly.

| Knob | **AI2 ref** (`7b_think_sft`) | **OLMo-core** (ours) | **Axolotl** (ours) |
|---|---|---|---|
| Engine / script | OLMo-core / `Olmo-3-7B-SFT.py` | OLMo-core / `Olmo-3-7B-SFT-local.py` (their script, beaker-stubbed; a **no-env run == AI2's recipe bit-for-bit**) | Axolotl + Transformers / YAML |
| Optimizer | SkipStepAdamW | SkipStepAdamW ✓ | `adamw_torch_fused` (no skip-step) |
| lr | 5e-5 | 5e-5 ✓ | 5e-5 (to apply) |
| epochs | 2 | 2 ✓ | 2 (to apply) |
| betas / wd / warmup / grad-norm | (0.9,0.95)/0/0.03→0/1.0 | ✓/✓/✓/✓ | ✓/✓/✓/✓ |
| global batch | 1,048,576 tok (fixed) | 1,048,576 ✓ (tune via `GLOBAL_BATCH_SIZE` / `RANK_MICROBATCH_TOKENS`, see [BATCHING.md](BATCHING.md)) | micro×accum (tune to match) |
| seq length | 32768 | **65536** (DEVIATION: no-truncation for our long-tail proof data, see [BATCHING.md](BATCHING.md) §1) | 32768 → 65536 (SP) |
| context-parallel `cp` | auto 2 @ 32768 | **auto 4 @ 65536** (cap 16384 tok/device) | `context_parallel_size` |
| Attention | **flash_2** (factory default) | flash_2 baseline; **flash_3 packed-varlen on the Hopper launcher** | flash_attention_3 (packed) / flex |
| Loss kernel | OLMo-core fused | fused ✓ | Cut Cross Entropy |
| RoPE / z-loss | YaRN / none | YaRN / none ✓ | model default / none |
| Context parallelism | auto `cp_degree=2` @ seq 32768 | **auto** ✓ (AI2's `BatchSizeConfig`; llama3 ring + doc mask) | `context_parallel_size` (ring) |
| FP8 *training* | **no — bf16** | **off by default (bf16)**, matching AI2 + NVIDIA Nemotron Nano (both post-train bf16, PTQ→FP8 only for inference); opt-in `OLMO_FP8=tensorwise\|rowwise` (keeps the 8 full-attention blocks' q/k/v/o + lm_head + embeddings high-precision). **Scaling is arch-gated** (cuBLAS FP8 GEMM, see below) | `fp8: true` opt-in (validate) |

**FP8 scaling by GPU arch** (verified `tools/fp8_probe.py`; `run.sh` auto-selects, explicit `OLMO_FP8` wins):

| arch | rowwise (accurate, ship) | tensorwise (lower-acc, A/B) | use |
|---|---|---|---|
| sm_90 Hopper / sm_100 B200 | ✅ | ✅ | **rowwise — production FP8** (validate on Hopper once BF16 is stable) |
| sm_120 RTX PRO 6000 | ❌ `CUBLAS_STATUS_NOT_SUPPORTED` | ✅ (fwd e4m3×e4m3 + bwd e5m2×e4m3) | **tensorwise — throughput comparison only** |

(`e5m2×e5m2` GEMMs fail on every arch and never occur in training — forward is e4m3×e4m3, backward grads are mixed e5m2×e4m3.) Caveat: torchao 0.15.0 vs torch 2.10 skips cpp extensions → FP8 scaling uses ATen fallback; bump torchao for real Hopper FP8 perf.

## Chat format & stop tokens (the ChatML alignment)

| | OLMo-core (ours) | Axolotl (ours) |
|---|---|---|
| Tokenizer | dolma2 (vocab 100278) | dolma2 (model tokenizer) |
| Chat template | `olmo` (open-instruct) | `olmo` (same jinja, `chat_template_jinja`) |
| End-of-turn (final) | `<\|endoftext\|>` (trained) — also the packer doc boundary | `<\|endoftext\|>` (trained) |
| Turn separator (multi-turn) | `<\|im_end\|>` | `<\|im_end\|>` |
| Masking | assistant content + eos trained; system/user/header masked | same (axolotl native) |
| Model `generation_config.eos_token_id` | **`[<\|im_end\|>, <\|endoftext\|>]`** (OLMo-core converter sets on HF export) | **`[<\|im_end\|>, <\|endoftext\|>]`** (set at export) |

Both ship `generation_config` stopping on **both** `<|im_end|>` and `<|endoftext|>`
— Olmo 3's own two-eos design — so the model is ChatML-inference compatible with no
data-format change. The **only** to-do is making sure the *exported* model's
`generation_config` lists both ids (OLMo-core's converter already does; for axolotl,
set it at export/serving).

## Genuine framework differences (not config lag)

1. **Optimizer**: OLMo-core's SkipStepAdamW skips loss-spike steps; axolotl's fused AdamW doesn't.
2. **Global batch**: OLMo-core/AI2 fix 1.05M tokens (grad-accum derived); axolotl sets micro×accum.
3. **Loss kernel**: OLMo-core fused vs axolotl CCE (numerically equivalent).
4. **Checkpoint**: OLMo-core needs distcp (HF→distcp conversion); axolotl loads HF directly.
5. **FP8**: axolotl-only prod option; not in AI2's bf16 reference.

## Implementation status
- **OLMo-core**: done — olmo template (open-instruct), recipe via AI2's CLI overrides.
  No format change needed (the two-eos generation_config makes it ChatML-servable).
- **Axolotl**: dev config aligned (olmo jinja, `<|endoftext|>`). To finish: align
  mid/prod to lr 5e-5 / 2 epochs + olmo jinja + `<|endoftext|>` (still on the old
  chatml/8e-5/3ep), and set the exported model's `generation_config.eos_token_id`
  to `[100265, 100257]`.
