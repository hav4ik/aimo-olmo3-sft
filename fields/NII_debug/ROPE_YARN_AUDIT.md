# RoPE / YaRN audit — training is correct; the conversion warning is a benign false alarm

**Date:** 2026-06-09. Triggered by a `[transformers]` warning seen during the 32B HF conversion:

```
RoPE scaling factor (config.rope_parameters['factor'] = 8.0) does not match the ratio implicitly set
by other parameters (implicit factor = post-yarn / pre-yarn = max_position_embeddings /
original_max_position_embeddings = -0.0001220703125). Using the explicit factor (8.0) in YaRN. ...
please correct the 'original_max_position_embeddings' fields in the model config.
```

We were worried this meant the **training** RoPE/YaRN was wrong (which would silently corrupt a ~$20k
run). It does **not**. Verified by 3 parallel code audits + the official HF configs + a direct
re-read of the training line and a reproduction of the warning.

## TL;DR
- ✅ **Training RoPE/YaRN is CORRECT** — the model trains with **exactly** the official Olmo-3 config.
  The weights are safe. This applies to **both** the 7B (already running on NII) and the 32B.
- ✅ **The exported `config.json` is also correct** in the FINAL artifact. The warning is a **transient,
  self-correcting** reload artifact (a `-1` placeholder for `max_position_embeddings` observed one line
  before it's patched to 65536). It changes **no weights** and **no shipped config field**.
- 🔧 No fix required for correctness. Optional: a post-export assertion in `upload.py` for peace of mind.

## 1. Training-time config (the $20k-critical part) — VERIFIED

Both production SFT scripts build the model identically (`with_rope_scaling`, applied to the
global/full-attention layers, not the sliding-window ones — matching Olmo-3's design):

```python
# olmocore/sft_scripts/Olmo-3-32B-SFT-local.py:533   (7B is byte-identical at Olmo-3-7B-SFT-local.py:516)
.with_rope_scaling(YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192))
```
with `rope_theta=500_000` (the `olmo3_7B` / `olmo3_32B` factory default — `OLMo-core/.../nn/transformer/config.py`).

`old_context_len` maps to HF's `original_max_position_embeddings`. **`factor × old_context_len = 8 × 8192
= 65536`** = the training `seq_len`. Internally consistent. The YaRN frequency blend
(`OLMo-core/.../nn/rope.py`, `compute_scaled_inv_freq`) depends only on `theta`/`dim`/`factor`/
`old_context_len` — **not** on `seq_len` — so there is no seq-len-vs-YaRN coupling that could drift. The
SFT script sets these **explicitly**, so it does not rely on (or get corrupted by) the base model's
`config.json`. The checkpoint is loaded as raw distcp tensors only (`LoadStrategy.never`); the base
HF config is never consulted to build the training graph.

## 2. Ground truth: the official HuggingFace configs
Fetched from `huggingface.co/allenai/Olmo-3.1-32B-Think/raw/main/config.json` and the 7B equivalent
(`allenai/Olmo-3-7B-Think`) — the RoPE block is **identical across both sizes**:
```json
"max_position_embeddings": 65536,
"rope_theta": 500000,
"rope_scaling": {
  "rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192,
  "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.2079441541679836
}
```

### Side-by-side — EXACT MATCH
| field | our training | official Olmo-3 config |
|---|---|---|
| `rope_theta` | 500000 | 500000 |
| `factor` | 8 | 8.0 |
| `original_max_position_embeddings` (`old_context_len`) | 8192 | 8192 |
| `beta_fast` / `beta_slow` | 32 / 1 | 32.0 / 1.0 |
| `attention_factor` | ≈1.20794 (`0.1·ln8 + 1`) | 1.2079441541679836 |

→ **We train with precisely the official Olmo-3 positional encoding. Weights are correct.**

## 3. Who does the weights conversion
A pipeline (both us and olmo-core):
1. **olmo-core's converter** does the heavy lifting — `OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py`
   (invoked as a subprocess by `fields/upload.py:177`) writes the HF safetensors **and** the initial
   `config.json`.
2. **Our `upload.py:120-128`** then post-normalizes the rope block (`rope_parameters → rope_scaling` +
   top-level `rope_theta`) for transformers-4.x / vLLM compatibility. It only **renames** the key and
   splits out `rope_theta`; it does **not** touch `original_max_position_embeddings` or
   `max_position_embeddings`.

## 4. Why the warning fires (and why it's benign)
Root-caused + reproduced in transformers 5.4.0:
- The dense Olmo-3 HF config builder writes **`max_position_embeddings = -1`** as a placeholder
  (`OLMo-core/.../nn/hf/config.py:128`). `original_max_position_embeddings` is written **correctly as
  8192** (`rope.py:281-290`, default `:244`).
- During conversion, `AutoConfig.from_pretrained(output_path)` (`convert_checkpoint.py:274`) reloads the
  config **while it still holds the `-1` placeholder** → transformers computes implicit factor
  `-1 / 8192 = -0.0001220703125` → the warning. (The warning text blames the wrong field — the real
  placeholder is `max_position_embeddings`, not `original_max_position_embeddings`.)
- **One line later** (`convert_checkpoint.py:275`) `max_position_embeddings` is patched to the real
  `max_sequence_length` (`upload.py` passes `-s 65536` at `:183`, default `:311`). So the **final
  on-disk `config.json` has `max_position_embeddings = 65536`** → implicit factor `65536/8192 = 8.0` =
  `factor` → correct, matches the official block in §2.
- **Weights are written before the config** (`convert_checkpoint.py:257`), so none of this touches tensors.

Reproduction: `max_position_embeddings = -1` → implicit factor `-0.000122…` (warns); `= 65536` → `8.0`
(no warning).

⚠️ **Caveat:** if you ever run `convert_checkpoint_to_hf.py` **directly without `-s 65536`**, you'll get
`max_position_embeddings = -1` in the *final* config too. **Always go through `upload.py`** (or pass the
seq_len), which supplies 65536.

## 5. Optional hardening (not required — final artifact is already correct)
Add a post-export assertion in `upload.py` (after the rope normalization) so a *future* converter change
that genuinely breaks the rope block fails loudly instead of shipping silently:
```python
rs = cfg["rope_scaling"]
assert rs["original_max_position_embeddings"] == 8192, rs
assert cfg["max_position_embeddings"] / rs["original_max_position_embeddings"] == rs["factor"], cfg
```
(Or, upstream-cleaner: olmo-core could set `max_position_embeddings = max_seq_len` at build time in
`nn/hf/config.py:128` instead of `-1`, as its hybrid path already does at `:384` — removes the noisy
round-trip. That's an OLMo-core change, not ours.)

## Verdict
**Training: correct (verified line-by-line vs the official config). Conversion/upload: final artifact
correct; the warning is a cosmetic, self-healing reload artifact.** Nothing to fix for correctness.

### Key references
- Training: `olmocore/sft_scripts/Olmo-3-32B-SFT-local.py:533`, `Olmo-3-7B-SFT-local.py:516`;
  `OLMo-core/.../nn/rope.py` (YaRN: `compute_scaled_inv_freq`, `to_hf_config` `:281-290`, `old_context_len` `:244`);
  `OLMo-core/.../nn/transformer/config.py` (`rope_theta=500_000`).
- Export: `OLMo-core/.../nn/hf/config.py:128` (`-1` placeholder), `convert_checkpoint.py:257` (weights),
  `:274-279` (reload-that-warns then patch); `fields/upload.py:120-128` (our mirror), `:177-186` (convert call), `:183` (`-s 65536`).
- transformers: `modeling_rope_utils.py:797-806` (the warning), `configuration_utils.py:461-466`
  (`rope_scaling` is an alias of `rope_parameters` in 5.4).
- Ground truth: `huggingface.co/allenai/Olmo-3.1-32B-Think` and `…/Olmo-3-7B-Think` `config.json`.
