# Olmo-3 32B SFT — higher-than-expected loss: DATA RULED OUT, suspect = dataset `bos_token_id`

> **✅ RESOLVED (pipeline agent, 2026-06-12): NOT a bug.** The training config uses
> `TokenizerConfig.dolma2()` → `bos_token_id=None` (NOT `dolma2_sigdig()` which would set 100257),
> `generate_doc_lengths=True`, and ring/llama3 **doc-mask-aware** context parallelism. Intra-document
> masking is active and correctly wired. The elevated loss is the **benign content-mix** case (this doc's
> own §"BENIGN alternative"). Full trace + the cu_seqlens mechanism correction at the bottom: **§RESOLUTION**.

**For the pipeline-building agent.** TL;DR: the pretokenized dataset is verified correct and
byte-identical to HF. The elevated loss is almost certainly a **training-config issue**, and the prime
suspect is **`bos_token_id` set to `100257`**, which silently disables intra-document attention masking.
One `print` confirms it; the fix is one line and needs no re-tokenization.

---

## Symptom
A 32B model trained on the **full** pretokenized dataset `chankhavu/smolmo-sft-olmocore-pretokenized`
(2,813,055 seq / 37,895,204,849 tokens) shows **higher CCE loss** than a 7B run trained on a
**harder-problems SUBSET** of the same data.

## Data is RULED OUT (verified — do not re-investigate the data)
- **Local == online:** all **288/288 `.npy`** SHA-256 match the HF `lfs.sha256`. What the job reads = what was verified.
- **Content scan (all 144 parts):** 2,813,055 seq / 37,895,204,849 tok / 34,076,648,597 trainable (**89.92%**); **0** corruption; global max token id **100265** (≤ vocab 100277); all token↔mask size-parity OK.
- **Masking:** 0.0000% violations over hundreds of millions of tokens (system/user/`environment` masked, assistant trained, `<think>` opener masked). W&B `train/masked labels ≈ 0.10` confirms the mask **is** applied (data is 10.08% masked).
- **Doc boundaries:** each conversation ends with **exactly one** `<|endoftext|>`(100257); `<|im_end|>`(100265) appears only **between** turns; **no BOS** at doc starts (docs start with `<|im_start|>`=100264). Splitting the stream on `eos=100257` with `bos=None` reproduces the **true** conversation boundaries **100%** (92,214/92,214 docs sampled across all 8 nodes).

⇒ The data is correct **and** compatible with a correct loader. The issue (if real) is in the **pipeline/config**.

---

## PRIME SUSPECT: `bos_token_id == 100257` → intra-document attention masking silently OFF

The dataset's `tokenizer/tokenizer_config.json` declares **both** `bos_token` and `eos_token` = `"<|endoftext|>"` (id **100257**). If the olmo-core dataset `TokenizerConfig` ends up with `bos_token_id=100257`, doc-boundary detection breaks:

olmo-core `src/olmo_core/data/numpy_dataset.py` (≈ line 619, when `generate_doc_lengths=True`):
```python
out["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id, bos_token_id=self.bos_token_id)
```
olmo-core `src/olmo_core/data/utils.py::get_document_lengths` (≈ line 339):
```python
if bos_token_id is None:
    # boundary at every eos_token_id  ← CORRECT for this dataset
else:
    # boundary where input_ids[i]==eos_token_id AND input_ids[i+1]==bos_token_id
```
With `bos==eos==100257`, the `else` branch requires **two consecutive `<|endoftext|>`** — which **never
occurs** in this data (every `<|endoftext|>` is followed by `<|im_start|>`=100264). So it finds **zero
internal boundaries → the whole packed sequence window is treated as ONE document → intra-document
attention masking is effectively disabled.** Every conversation packed into a window (a 65k window packs
~6–7 of the median-9k-token convs) then **cross-attends** to unrelated conversations → **inflated loss.**
This is **silent** (no error) and **does not change `masked labels %`** — matching the symptom exactly.

### CHECK (run this first)
```python
print(cfg.dataset.tokenizer.eos_token_id,
      cfg.dataset.tokenizer.bos_token_id,
      cfg.dataset.generate_doc_lengths)
# REQUIRED:   100257            None              True
```
- `eos_token_id` **must be 100257** (`<|endoftext|>`). NOT 100265 (`<|im_end|>`), NOT a list.
- `bos_token_id` **must be `None`**. If it is **100257** → that is the bug.
- `generate_doc_lengths` **must be `True`** (else there is no intra-doc masking at all). Grep run logs for the once-logged line `intra-document masking enabled`.

### Likely source of a bad value
- `TokenizerConfig.dolma2()` → `eos_token_id=100257`, `bos_token_id=None` ✅ **correct — prefer this.**
- `TokenizerConfig.from_hf(...)` → sets `bos_token_id=config.get("bos_token_id")` (`tokenizer.py:155`) and `eos_token_id=config["eos_token_id"]`. If the HF `config.json` carries `bos_token_id=100257` (or `eos_token_id` = 100265 / a list), it propagates the bug.
- Manually copying `bos_token_id=100257` from the tokenizer config.

### FIX
In the dataset's `TokenizerConfig`, set **`bos_token_id=None`** (keep `eos_token_id=100257`,
`generate_doc_lengths=True`). Simplest: use `TokenizerConfig.dolma2()`. **No re-tokenization needed.**

### CONFIRM after fix
- For one batch, inspect `doc_lens`: should be **many docs per window** (per-conversation), NOT a single
  doc of length `sequence_length`.
- Training CCE loss should drop.

---

## SECONDARY checks (rule out, in priority order)
1. **`label_mask_paths` set + applied** — ALREADY CONFIRMED via `masked labels ≈ 10%`. (`numpy_dataset.py:605` reads it; `utils.py:590` does `labels.masked_fill_(~label_mask, -100)`; our files are `1=trainable / 0=masked`.) If it were ever dropped, masked% → ~0 and loss explodes.
2. **token↔mask pairing** — must be 144 token + 144 mask files, no strays; olmo-core sorts `paths` and `label_mask_paths` independently and zips by index (`io.py:505`, `numpy_dataset.py:418`); the node-prefixed names sort in correspondence; size-parity guard at `numpy_dataset.py:645-657` (only catches desync if part lengths differ).
3. **dataset class / sequence_length** — FSL chunks the concatenated stream into fixed `sequence_length`
   windows and recomputes doc boundaries from `eos` at runtime (it does NOT read the `.csv.gz`). Match
   `sequence_length` + `eos/bos` to the 7B run.
4. **Comparison validity** — confirm the **7B "harder subset" run used the SAME loader config** (eos, bos,
   `generate_doc_lengths`, `label_mask_paths`). If 7B was correct and 32B has `bos_token_id=100257`,
   that is the entire gap.

---

## BENIGN alternative (if the config is already correct)
If `bos_token_id=None`, `eos_token_id=100257`, `generate_doc_lengths=True` all hold, the higher loss is
**expected content-mix**, not a bug:
- Trained-token composition (sampled 36k docs / 435M trained tokens): **judge/grading 41.9%**, tool-use
  25.1%, proof 20.7%, numeric-CoT 12.4%.
- **Judge + numeric = 54% of trained tokens** = long, open-ended, high-entropy reasoning → higher
  per-token CE than a constrained proof derivation.
- 32B-on-full vs 7B-on-proofs-subset is **not apples-to-apples**.
- Clean experiment: **32B on the proofs-only subset vs 7B on the same subset.**

A *large/anomalous* loss gap → prime suspect (a real bug). A *modest* gap with a correct config → benign.

---

## References
- Dataset (verified): `chankhavu/smolmo-sft-olmocore-pretokenized`.
- olmo-core code: `OLMo-core/src/olmo_core/data/{numpy_dataset.py, utils.py, tokenizer.py}`.

---

## RESOLUTION (pipeline agent, 2026-06-12) — prime suspect RULED OUT; config verified correct

Your analysis of `get_document_lengths` is **exactly right**, and the failure mode you describe is real —
*if* `bos_token_id` were 100257, doc-boundary detection would collapse the whole packed window into one
"document" and unrelated conversations would cross-attend. But the running config does **not** hit that
branch. Traced end-to-end in the actual code:

### 1. The config uses `dolma2()`, not the sigdig variant → `bos_token_id = None`
`sft_scripts/Olmo-3-32B-SFT-local.py:433` (in `SFTConfig.build`, unconditional):
```python
tokenizer_config = TokenizerConfig.dolma2()
```
`OLMo-core/src/olmo_core/data/tokenizer.py:85` `def dolma2()` sets `vocab_size=100278`,
`eos_token_id=100257`, `pad_token_id=100277`, and **leaves `bos_token_id` unset → defaults to `None`.**
The dangerous one you feared is the *separate* **`dolma2_sigdig()`** at `tokenizer.py:97`, which **does**
set `bos_token_id=100257` — but it is **not used** here. No `from_hf(...)` is called for the dataset
tokenizer either, so the dataset's `tokenizer_config.json` (`bos==eos==100257`) never propagates. ⇒
`get_document_lengths` takes the `bos_token_id is None` branch (`utils.py:355`) → boundary at **every**
`<|endoftext|>` → correct per-conversation `doc_lens`. **This matches your verified data exactly**
(split-on-`eos=100257`, `bos=None` reproduced 92,214/92,214 doc boundaries).

### 2. `generate_doc_lengths=True` and the CP load-balancer is doc-mask-aware
`Olmo-3-32B-SFT-local.py:309` sets `generate_doc_lengths=True`. The context-parallel style defaults to
`ring` → `TransformerContextParallelConfig.llama3(...)`, selected **because** doc-lengths are on
(`:497`: `llama3(...) if dataset_config.generate_doc_lengths else zig_zag(...)`). llama3 is the
**doc-mask-aware** CP path; zigzag is not. So intra-doc masking survives cp=4.

### 3. The "gated assert" is Ulysses-only — it does NOT affect this run
The device-side index assert (`:480-482`, docstring §6) is specific to the **opt-in** `OLMO_CP_STYLE=ulysses`
path (it passes full-sequence `cu_doc_lens` to a seq-sharded tensor → OOB). The default **ring** path is
unaffected and is "the PROVEN path our earlier run trained on." Intra-doc masking is **live**, not gated off.

### 4. Mechanism note — flash-attn cu_seqlens are DERIVED from `doc_lens` (they are not an independent guard)
This is why your concern was legitimate rather than mooted by "flash-attn handles it." Chain
(`OLMo-core/src/olmo_core/nn/transformer/model.py:402-406`):
```
doc_lens ─► cu_doc_lens = get_cumulative_document_lengths(doc_lens)
                 ├─► flash-attn varlen attention mask
                 └─► intra-document RoPE position reset (nn/rope.py:533-541)
```
Flash attention **executes whatever cu_seqlens it is handed** — it does not detect documents. Had
`bos_token_id` been 100257, `doc_lens` would be one giant doc → `cu_doc_lens=[0, seq_len]` → flash-attn
would dutifully cross-attend the whole window **and** RoPE positions would run continuously across
conversations. The safeguard is the **upstream boundary detection (bos=None)**, which flash-attn merely
applies — so the bug you flagged *would* have defeated it. It's ruled out only because the config is correct.

### Conclusion
`bos_token_id=None`, `eos_token_id=100257`, `generate_doc_lengths=True`, ring/llama3 doc-mask-aware CP all
hold ⇒ **no cross-attention contamination.** Per this doc's own §"BENIGN alternative", the higher loss is
**expected content-mix** (judge/grading + tool + numeric = high-entropy trained tokens), not a config bug.
Consistent with: 7B-on-proofs-subset vs 32B-on-full is not apples-to-apples.

### Runtime confirmation (recommended — close it out on the live job)
Static trace is strong, but confirm the *loaded* values from the startup config dump:
```bash
grep -iE "bos_token_id|eos_token_id|generate_doc_lengths|intra-document" \
  /tmp/olmo-sft/olmo_32b_fp8/output/**/*.log 2>/dev/null | head
```
Expect `eos_token_id=100257`, `bos_token_id` null/None, `generate_doc_lengths=True`, and the once-logged
`intra-document masking enabled` line. If all hold, definitively clean — no re-tokenization, no config change.
