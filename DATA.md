# Data & prompting — audit reference

The SFT data recipe shared by both frameworks (`axolotl/` + `olmocore/` in this repo).
**Principle:** data is the most important part of the pipeline, so we keep the
custom layer thin and lean on each framework's own machinery. This file shows exactly
what the model sees, token by token, so it can be audited.

> **NOTE (current):** since switching the base to `Olmo-3-7B-Think`, the template is
> **think-aware** — axolotl uses `chat_template: tokenizer_default` (the model's own
> `chat_template.jinja`) and OLMo-core uses open-instruct's `olmo_thinker` (via
> `data_prep/prepare.sh --template`). The token-level examples below were written for the
> generic `olmo` template; they still illustrate the MASKING PRINCIPLE (assistant content +
> eos trained, system/user/headers masked), but the exact think-template bytes differ.

## Source of truth: OLMo's own stack

Olmo 3 was post-trained with **AI2's open-instruct + OLMo-core**, so we stay true
to that stack rather than hand-rolling tokenization:

- **OLMo-core side**: tokenized by open-instruct's canonical
  `scripts/data/convert_sft_data_for_olmocore.py` with its registered **`olmo`**
  chat template, run in a dedicated data-prep image
  (`open-instruct/Dockerfile.dataprep`) via `data_prep/prepare.sh`. This
  is the exact tool/template Olmo 3 was trained with → correct masking and the
  OLMo-core `.npy` format, no custom code.
- **axolotl side**: uses axolotl's native `chat_template` machinery with
  `chat_template: tokenizer_default` → the model's own think-aware `chat_template.jinja`
  (shipped with `Olmo-3-7B-Think`). axolotl computes the mask itself.

Both therefore use the **same olmo template** and the **same masking policy**,
verified below to agree token-for-token.

## The `olmo` chat template

- Injects a default Olmo **system** prompt if none is present.
- Wraps each turn `<|im_start|>{role}\n{content}<|im_end|>\n`.
- **Ends the conversation with `{{ eos_token }}` = `<|endoftext|>`** (id 100257)
  instead of `<|im_end|>`. This is Olmo 3's real stop token (verified:
  `Olmo-3.1-32B-Think`'s tokenizer reports `eos_token='<|endoftext|>'`), and the
  single eos OLMo-core's packer splits documents on. `<|im_end|>` (100265) is
  only the intra-conversation turn separator.

| token | id | role |
|---|---|---|
| `<\|im_start\|>` | 100264 | turn header start (masked) |
| `<\|im_end\|>` | 100265 | turn separator (masked) |
| `<\|endoftext\|>` | 100257 | **eos** — conversation end / stop / doc boundary (TRAINED) |

## Worked example (real row from `allenai/tulu-3-sft-personas-math`)

Single-turn `[user, assistant]`. Rendered (open-instruct `--visualize` + axolotl
preprocess agree):

```
<|im_start|>system
You are OLMo, a helpful function-calling AI assistant built by Ai2. … <functions></functions><|im_end|>
<|im_start|>user
A young student studying architectural design …<|im_end|>
<|im_start|>assistant
### Problem Solution … I hope it is correct.<|endoftext|>
```

Label mask (`TRAIN` = loss; `--` = masked), from axolotl's prepared dataset:

```
  --    198  'Ċ'
  -- 100264  '<|im_start|>'   \
  --  78191  'assistant'       |  assistant HEADER: MASKED
  --    198  'Ċ'             /
TRAIN  10267  'Let'            \
TRAIN    596  "'s"             |  assistant CONTENT: TRAINED
  ...                          /
TRAIN 100257  '<|endoftext|>'  <- eos / stop: TRAINED
```

System + user turns and both `<|im_end|>` separators are masked; the assistant
**header is masked**, content + final `<|endoftext|>` are trained.

## Cross-framework masking agreement (audited)

| signal | open-instruct (OLMo-core) | axolotl |
|---|---|---|
| trainable fraction (corpus / example) | 73.3% | ~64–75% |
| `<\|endoftext\|>` trained | yes (1/conv) | yes (1/1) |
| `<\|im_end\|>` trained | no | no (0/2) |
| `<\|im_start\|>` / assistant header | masked | masked |

Both mask the assistant header (my earlier hand-rolled prep wrongly trained it —
removed). The small trainable-fraction spread is per-example length variation,
not a masking difference.

## How each framework consumes the data (offline tokenization; training never tokenizes)

- **OLMo-core**: `prepare.sh` (open-instruct) → `token_ids_part_*.npy` +
  `labels_mask_part_*.npy` (offline). `Olmo-3-7B-SFT-local.py` memory-maps the `.npy`;
  it never tokenizes. Packer splits documents on `<|endoftext|>`.
- **axolotl**: `axolotl preprocess` tokenizes once into `dataset_prepared_path`
  (offline) with the olmo template; `axolotl train` reads that cache. That cache is
  **uncompressed arrow (~120 GB for our 6 B-token set)** and not portable. To ship a
  compact, pushable pre-tokenized form (~15 GB) that Axolotl loads natively, see
  **[AXOLOTL_PRETOKENIZED.md](AXOLOTL_PRETOKENIZED.md)** + `data_prep/arrow_to_pretokenized_parquet.py`.

Note: open-instruct **truncates** long examples to `--max_seq_length`; axolotl
**drops** examples longer than `sequence_len`. So keep `sequence_len` ≥ the prep's
max length (math solutions avg ~1335 tok — 512 drops nearly all; use 2048+).

## Recipe hyperparameters (from OLMo-core upstream — the reference)

From `OLMo-core/src/scripts/train/sft/Olmo-3-{7B,32B}-SFT.py` (identical):
**`SkipStepAdamW`** (lr `8e-5`; README launch example overrides to `5e-5`), betas
`(0.9, 0.95)`, weight_decay `0.0`, eps `1e-8`; `LinearWithWarmup` warmup `0.03`
→ `0`; `max_grad_norm 1.0`; `z_loss=None`; **3 epochs**; global batch = **64
sequences**; seq 16k (32k recommended); selected-module activation checkpointing;
YaRN rope (factor 8). The OLMo-core trainer uses `SkipStepAdamW` directly; axolotl
conforms with the same lr/betas/wd/schedule/grad-norm (its nearest optimizer is
`adamw_torch` — no skip-step wrapper).
