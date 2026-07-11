# Verification of the SHIPPED pre-tokenized dataset

Record of the final checks on `chankhavu/smolmo-sft-olmocore-pretokenized/olmocore/` (the actual
uploaded `.npy`, downloaded back and verified — i.e. what the trainer consumes, not just what the
converter *could* produce).

## 1. Coverage / completeness (HF metadata)
- All **8 nodes** present (prefixes 000–007), each **18 token_ids + 18 labels_mask + 18 csv.gz + 1 stats**; tokenizer/ present; total 189.5 GB.
- **126→144 token/mask pairs: token_ids bytes == 4 × labels_mask bytes**, 0 mismatches (olmo-core's size-parity requirement; uint32 vs bool).

## 2. Full content scan (read every one of the 144 parts)
Per-node actual token count == that node's `stats_part_*.json` (all 8 match exactly). Grand totals:

| metric | value | target | match |
|---|---|---|---|
| sequences | 2,813,055 | 2,813,055 | ✅ exact |
| tokens | 37,895,204,849 | 37,895,204,849 | ✅ exact |
| trainable | 34,076,648,597 (89.92 %) | — | — |
| global max token id | 100265 (`<|im_end|>`) | ≤ 100277 | ✅ in range |
| size/len mismatches | 0 | 0 | ✅ |

## 3. Adversarial audits (sub-agent councils) — all GREEN
- **Verification-methodology audit:** the count/size/max-id checks prove *lengths* only, so the missing
  *value* probes were run — 38k–44k distinct token ids/part (not garbage), zeros 0.04 % (benign `!`),
  special tokens at plausible rates, masks never all-0/all-1 (trainable banded 0.898–0.902), **all 144
  csv boundaries partition `[0, part_len)` cleanly**, per-node token+trainable totals reconcile exactly.
  Verdict: **no corruption**; node 006 (slow path) structurally identical to fast-path nodes.
- **Format/masking audit:** decoded ~3,100 docs across all 8 nodes — structure, `<think> … </think>`,
  positional terminators, and masking all **PASS**; tool examples (`<functions>` masked in system,
  `<function_calls>` trained in assistant, `environment` masked) **PASS**. 0 malformed.

## 4. Manual tool-use spot-check (from the shipped `.npy`)
Decoded a real multi-turn tool doc (part `000_0000`, tokens 106399:107971) straight from the downloaded
`token_ids`/`labels_mask` `.npy`:

```
system        trained 0.000  ends <|im_end|>   masked   (incl. <functions> declaration)
user          trained 0.000  ends <|im_end|>   masked
assistant     trained 0.991  ends <|im_end|>   TRAINED  (makes the tool call)
environment   trained 0.000  ends <|im_end|>   masked   (tool output)
assistant     trained 0.990  ends <|endoftext|> TRAINED (final answer)
```
- `<think>` boundary: `<th`(0) `ink`(0) `>`(0) | ` We`(1) — clean masked→trained cut at the space.
- assistant tool call `<function_calls>stateful_python_code_exec(code='…')` → **all mask=1 (TRAINED)**,
  closing `</function_calls><|im_end|>` trained.
- `<|im_start|>environment\n …tool output…` → **all mask=0 (MASKED)**.
- final answer `…\boxed{…}<|endoftext|>` → trained.

Matches the agreed spec exactly: train the model's outputs (reasoning, tool calls, answers); mask
everything it shouldn't predict (system, user, tool outputs, the gen-prompt opener).

## Conclusion
The shipped dataset is **complete, uncorrupted, and correctly formatted/masked** — verified from the
actual uploaded tensors. Training-ready.
