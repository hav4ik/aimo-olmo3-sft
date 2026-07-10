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

### Ulysses + intra-doc RoPE — FIXED (olmo-core `c7cfa2a`)

Ulysses shards the sequence contiguously and passes the FULL `cu_doc_lens` through, but RoPE runs on the
shard (before the all-to-all). Previously `rope.forward` re-derived per-document positions from a LOCAL
`flat_idx` against the FULL `cu_doc_lens`, so any rank whose shard didn't begin at global position 0 got
**wrong RoPE phases** — corrupting logits and tripping a device-side index assert under torch.compile
(`index out of bounds: 0 <= idx < max_tokens_per_rank`). This blocked long-context (cp≥2) sink training.

**Fix (Ulysses-only; ring path unchanged):** bake the per-document position reset into the RoPE buffers
BEFORE the load balancer shards them (`Transformer._prepare_inputs`, gated on `UlyssesLoadBalancer`), and
apply the pre-sharded buffers per-row in `Attention.forward` (pass `cu_doc_lens=None` to RoPE). Ring still
re-shards `cu_doc_lens` per rank and resets inside `rope.forward` as before.

Verified: isolated RoPE fp32 exact (Δ=0), full-model Ulysses + 5 doc layouts == non-CP (delta = bf16
reduction noise, matching the no-docmask CP baseline), ring result identical before/after. Test:
`src/test/nn/transformer/cp_intra_doc_test.py` (Ulysses, FA2 + FA3, run on 2+ GPUs). Ring is skipped
(zig-zag gather; upstream already skips ring CP; ring rejects sinks anyway).

So: **long-context sink training now works on Ulysses.** Ring is still not an option with sinks. cp=1 is
also fine (no CP) for `seq_len <= max_tokens_per_rank`.

## Long-context on the RTX 6000: memory + compile

Two RTX-6000-specific walls after the CP fix (both tunable, neither a correctness issue):

**1. Device OOM.** 32B + AdamW is ~64 GB/rank (FSDP-sharded), leaving little for 65536-token activations
(80 GB H100 → ~16 GB; 96 GB RTX 6000 → ~32 GB). Knobs, in order of impact:
- `--ac-budget` LOWER = recompute more = less memory (1=recompute nothing/most mem, 0=recompute all/least).
  You want ~0.1 for long context, not 0.8. (AI2: 32B=0.3, long-ctx SFT=0.1.)
- `--max-tokens-per-rank 8192` → cp8 (all 8 GPUs in CP): half the activation memory of cp4. (32B has 40
  heads, 40%8=0 ✓.)
- If cp8 + `--ac-budget 0` still OOMs, the 64 GB optimizer floor is the wall → 8-bit AdamW or multi-node.

**2. Triton shared-memory OOM at compile** (`No valid triton configs / out of resource: shared memory`,
Required 196680 > limit ~101376). The wide RMSNorm (over the 5120 hidden dim), fused with the residual
add, compiles to a *persistent* reduction that loads the full row into shared memory (~196 KB).
**Hopper/B200 (~228 KB smem/block) fit it; the RTX 6000 (~99 KB) and A100 (~163 KB) do not.**

⚠️ Setting `torch._inductor.config.triton.persistent_reductions=False` is **NOT sufficient** — for the
fused (residual-add + norm) block op inductor still emits a persistent reduction and compile dies anyway
(confirmed on the RTX 6000). **The real fix: `fused_rms`.** The sft script auto-swaps *all* RMSNorm sites
to `FusedRMSNorm`, which calls flash-attn's hand-tiled Triton `rms_norm` kernel (no giant smem tile)
instead of inductor codegen — same fp32 RMSNorm math, and the path olmo-core's own `fused_ops=True` uses
(so it's FSDP2 + compile safe).

The wide (d_model=5120) norms are `block.layer_norm`, `lm_head.layer_norm` **and `q_norm`** — note
`olmo3_32B` sets `use_head_qk_norm=False`, so `q_norm` normalizes the *full* `n_heads*head_dim = 5120`
projection (NOT per-head/head_dim), i.e. it's exactly as wide and would OOM the same way if left on `rms`.
All three (plus `k_norm`@1024) share one `LayerNormConfig`, so the swap `replace`s it once and points every
slot at the fused config. Auto-on when `shared_memory_per_block_optin < 200 KB`; Hopper/B200 keep stock
`rms`. Force with `--fused-rmsnorm 1/0` (`OLMO_FUSED_RMSNORM`, tolerant of `1/true/yes/on` vs
`0/false/no/off`). The `persistent_reductions` flag is still flipped as defence-in-depth. Eager fallback
(no compile at all): `-e TORCHDYNAMO_DISABLE=1` (drop `--ac-budget`; budget mode needs compile). Tests:
recipe `olmocore/tests/test_fused_rmsnorm_swap.py` (swap logic + all-wide-norm coverage + build), olmo-core
`layer_norm_test.py::test_fused_rms_norm_wide_matches_rms_fwd_bwd` (fwd/bwd parity at 5120).
