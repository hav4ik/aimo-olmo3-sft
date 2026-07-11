# Masking-correctness check (role alignment) — beyond equivalence-to-stock

The byte-diff (RESULTS.md) proves patched==stock, but both run the SAME open-instruct `mask_labels`,
so it cannot catch a latent masking bug. This check verifies the masks actually land on role
boundaries for our real data — i.e. that open-instruct's prefix-rendering mask logic is correct here.

`mask_labels` computes mask spans by tokenizing message PREFIXES separately and slicing the FULL
tokenization at those lengths. That requires prefix-stability (tok(render(messages[:k])) is a
token-prefix of tok(render(full))), which is NOT generally true for BPE. It holds here because:
- `<|im_start|>`(100264)/`<|im_end|>`(100265) are atomic special tokens → hard, prefix-stable walls
  at every turn seam;
- the one soft seam (generation header `<|im_start|>assistant\n<think>`) is made prefix-stable by the
  `<think> ` (space) design: `>` stays token 29 whether end-of-prefix or followed by ` <word>`.

Evidence (validate/role_align.py on out_glob, real shards):
- SEQUENCE 0 (tool-use, 7 rounds): system/user MASKED, assistant TRAINED, every `environment`
  (tool output) MASKED, every assistant turn TRAINED. Mask boundary at first assistant turn:
  `<th`/`ink`/`>` (the `<think>` header) MASKED, first reasoning token ` We` TRAINED — clean cut.
- SEQUENCE 1 (single-turn): system/user MASKED, assistant TRAINED; same clean `<think>` boundary.

Conclusion: masking is correct for our data (environment masked, assistant trained, gen-prompt opener
masked, byte-clean boundary). This is upstream open-instruct logic — UNCHANGED by our patch.
