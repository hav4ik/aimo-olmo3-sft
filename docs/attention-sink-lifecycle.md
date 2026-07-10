# Attention-sink lifecycle (Olmo-3 32B SFT, cu128 image)

How per-head attention sinks are handled end-to-end in this container
(`chankhavu/olmo3-olmocore:cu128-fa2-sink`), what's verified vs reasoned, and the gotchas found
getting `chankhavu/yccchen-olmo3-deploy` to train.

## TL;DR

| Stage | Where | Status |
|---|---|---|
| **Read** HF → olmo-core | `OLMo-core/src/olmo_core/nn/hf/checkpoint.py::load_hf_model` | ✅ verified |
| **Use** in the forward | `nn/attention/*` (post-correction on FA2/FA3) | ✅ verified (unit tests) |
| **Train** the sinks | sft script optimizer (all params) | ✅ structurally correct; confirm values move |
| **Write** distcp checkpoint | olmo-core distributed checkpoint | ✅ automatic (sinks are a Parameter) |
| **Write** HF export | `checkpoint.py::save_hf_model` | ✅ preserves sinks; ⚠️ not `trust_remote_code` format |

The sink is a per-head learnable scalar logit (`self_attn.sinks`, shape `[num_attention_heads]`, one
per layer). Same mechanism as gpt-oss / Yi-Chia's `Olmo3Sink`.

---

## The model: `chankhavu/yccchen-olmo3-deploy`

It is saved as a **stock `olmo3`** checkpoint (`model_type: olmo3`, `architectures:
["Olmo3ForCausalLM"]`, **no `auto_map`**) with the trained `self_attn.sinks` carried as **extra
tensors**. It is *not* in Yi-Chia's `trust_remote_code` `Olmo3Sink` format. Consequence: plain
`transformers.AutoModelForCausalLM` loads it as stock `Olmo3ForCausalLM`, which has no `sinks`
parameter and **silently drops all 64 layers' trained sinks** (the `UNEXPECTED: self_attn.sinks`
report). That report is printed by transformers itself and always appears — the fix is what happens
after it.

---

## 1. Read (HF → olmo-core) — `load_hf_model`

`load_hf_model` loads the HF model through a **sink-aware class** so the trained sinks come in:

1. **`auto_map` present** → `trust_remote_code=True` (the model's own class).
2. **raw weights carry `*.sinks`** → a minimal `Olmo3SinkForCausalLM` (subclass of the *installed*
   transformers `Olmo3ForCausalLM` + a `sinks` Parameter) loads them natively. Look for
   `Loaded via sink-aware Olmo3SinkForCausalLM`.
3. else stock.

**Safety net:** `_recover_dropped_sink_weights` re-reads `*.sinks` straight from the raw safetensors
after a stock load. If you see `Recovered 64 attention-sink tensor(s) from raw safetensors`, the
fallback did the job — still correct.

The stock deploy artifact has no `auto_map`, so on it you'll see either the sink-aware-class line or
the recovery line; both put all 64 trained sinks into the converted distcp checkpoint.

## 2. Use (forward) — `nn/attention`

The sink is applied via **exact post-correction** on both FA2 and FA3:
`o_sink = o / (1 + exp(sink − lse))`, overwriting flash's saved `(o, lse)` so the native backward gives
exact `dq/dk/dv`, plus a Triton `dsink = −Σ p_sink·δ`. This equals Yi-Chia's eager reference (append the
sink as an extra softmax logit, drop after normalization). **In-kernel FA3 is NOT active on this image**
— see gotchas. Verified in `src/test/nn/attention/attention_sink_flash_test.py` (FA2 `varlen=True` is the
production path on the RTX 6000).

## 3. Train the sinks

`sinks` is an `nn.Parameter` (built with `use_sink=True`), so it's in `model.parameters()` and trained
by the optimizer with everything else (matches Yi-Chia's `AdamW(model.parameters())`). The `dsink`
gradient path is unit-tested. **Not yet confirmed empirically on a real run** — verify the values move:

```bash
# sinks in the INPUT model
python - <<'PY'
from safetensors import safe_open; import glob
for f in glob.glob('/data/training/hf_models/chankhavu/yccchen-olmo3-deploy/*.safetensors'):
    with safe_open(f,'pt') as h:
        ks=[k for k in h.keys() if k.endswith('.sinks')]
        if ks: print('input', ks[0], h.get_tensor(ks[0])[:4].tolist()); break
PY
# after training, unshard the trained checkpoint and compare — they should differ.
```

## 4. Write distcp checkpoint

Automatic: `sinks` is a registered Parameter, so olmo-core's distributed checkpoint saves it (and its
optimizer state) with the rest of the model. No special handling.

## 5. Write HF export (olmo-core → HF) — `save_hf_model`

When the converted state has `*.sinks`, `save_hf_model` saves through the sink-holder subclass so the
trained sinks are preserved as extra tensors → **stock `olmo3` + sinks, the same format as the input**
`yccchen-olmo3-deploy`. Any sink-aware loader (this converter, or Yi-Chia's class) reads them back.
Verified round-trip on a synthetic model (save → reload → sinks match). **Not yet run on a real
trained 32B checkpoint.**

⚠️ This is **not** Yi-Chia's `trust_remote_code` `Olmo3Sink` format (`model_type=olmo3_sink` +
`auto_map`). Her `modeling_olmo3_sink.py` targets an older transformers rope API and **cannot be
constructed under this image's transformers 5.13** (`Olmo3SinkRotaryEmbedding` →
`rope_parameters[layer_type]` `KeyError: None`) — the same reason `trust_remote_code` falls back on
load. A full `auto_map` export needs sink modeling written for transformers 5.13.

---

## Gotchas found (all fixed unless noted)

- **transformers drops sinks** on stock `Olmo3ForCausalLM` load AND `from_config`+`load_state_dict`
  save → sink-aware class load + recovery (read) and sink-holder save (export).
- **In-kernel FA3 not available**: torch 2.10 compiles flash-attn 3's stable-ABI `flash_api_stable.cpp`,
  but the sink patch only edits `flash_api.cpp` (torch < 2.9.0.dev). FA3 runs **post-correction**
  (exact, numerically identical for a fresh SFT). See `fa3-inkernel-needs-old-torch` note.
- **torchaudio ABI**: base pinned torch 2.10 but pulled torchaudio 2.11 → `undefined symbol
  torch_dtype_float4_e2m1fn_x2`; transformers imports it loading olmo3 → convert crashed. Removed
  torchaudio in the deploy layer.
- **Her modeling code ≠ transformers 5.13** (rope API): imports but won't construct.

---

## Known smoke pitfall: NaN loss at step 1 with `--seq-len 128`

A 128-token smoke on a packed 256K SFT dataset can NaN at step 1: with a small `--gbs`, each rank gets
~1 short sequence, and the first 128 tokens of a packed example are **all prompt (label-masked)** → the
cross-entropy averages over **0 valid tokens** → NaN. This is a **degenerate-smoke artifact**, not a
training bug (the real `--seq-len 65536` run has completion tokens). Fixes:

- Smoke at a larger length so completions appear, e.g. `--seq-len 4096` (and raise `--gbs` so each rank
  gets ≥2 sequences), or
- Go straight to the real run: `--seq-len 65536 --epochs 2 --max-tokens-per-rank 16384` (ring CP — see below).

To isolate a *real* forward NaN from the masking artifact: rerun the smoke with `--sink 0`. If the NaN
persists it's not the sinks (batch/data/precision); if it clears, investigate the loaded sinks.

## Context parallelism + sinks — the hard constraint

The sink post-correction needs the **complete** softmax `lse` for each query. That drives which CP
styles work:

| CP style | Sink? | Why |
|---|---|---|
| **cp=1 (no CP)** | ✅ | flash computes the complete local lse |
| **Ulysses** (all-to-all) | ✅ | each rank gets the FULL sequence for its head-slice → complete lse |
| **Ring** | ❌ **rejected** (`_reject_ring_sink`) | ring builds the lse incrementally across ranks; the per-head sink correction can't be applied |

So a sink model at **cp≥2 requires Ulysses** — ring will `raise`.

**BUT Ulysses currently trips a device-side index assert** with cp≥2 + intra-doc masking + torch.compile:
`index out of bounds: 0 <= idx < <max_tokens_per_rank>`. Ulysses passes the FULL-sequence `cu_doc_lens`
while the position/bucketize path sees a `max_tokens_per_rank`-sharded tensor → OOB. See the
`OLMO_CP_STYLE` note in `Olmo-3-32B-SFT-bf16.py`.

**Net:** long-context (cp≥2) sink training is blocked until the Ulysses OOB is fixed. Until then:
- **cp=1**: `--seq-len <=max_tokens_per_rank>` (e.g. `--seq-len 16384 --max-tokens-per-rank 16384`) —
  sink works, no CP. Fits the RTX 6000 (same per-rank memory as 65536/cp4).
- **cp≥2 / 65536+**: fix the Ulysses `cu_doc_lens`-vs-shard OOB (TODO), or go multi-node.
- Ring is **not** an option with sinks.
