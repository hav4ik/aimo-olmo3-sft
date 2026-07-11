# smolmo OLMo-core pre-tokenization — where do I find what

Master index for the pre-tokenized SFT dataset for **Olmo-3 Thinking** training, the fast converter
that produced it, and all the validation/audit records.

---

## 1. The data — where it lives

### ✅ FINAL pre-tokenized dataset (train on this)
- **HF:** `chankhavu/smolmo-sft-olmocore-pretokenized` → the **`olmocore/`** folder (≈189.5 GB)
  - `token_ids_part_{000-007}_{0000-0017}.npy` (uint32, raw memmap — read with `np.fromfile(dtype=np.uint32)`, **not** `np.load`)
  - `labels_mask_part_*.npy` (bool, 1 = trainable / 0 = masked; position-aligned with the same-named token_ids)
  - `token_ids_part_*.csv.gz` (gzip; `start,end` document token-ranges within each part — olmo-core ignores these, it recomputes offsets from file sizes)
  - `tokenizer/` (the Olmo-3-7B-Think tokenizer + `chat_template.jinja`)
  - `dataset_statistics.json` (merged stats)
- **Stats:** 2,813,055 sequences · 37,895,204,849 tokens · 34,076,648,597 trainable (**89.92 %**) · 0 filtered · 144 parts (8 nodes × 18).
- **Local mirror (deletable backup):** `/mnt/data/olmocore-verify/olmocore/` (177 GB, byte-identical to HF).

**Train against:**
```python
paths            = "hf://datasets/chankhavu/smolmo-sft-olmocore-pretokenized/olmocore/token_ids_part_*.npy"
label_mask_paths = "hf://datasets/chankhavu/smolmo-sft-olmocore-pretokenized/olmocore/labels_mask_part_*.npy"
```

### Source SFT dataset (human-readable conversations)
- **HF:** `chankhavu/smolmo-sft-v2-seqlen64k` (≈35.9 GB, 1024 `data/train-*.parquet` shards; each row = a `messages` conversation).
- **Local:** `/mnt/data/proof-redesign/smolmo-math-cot-sft/` (`data/` shards + `README.md` dataset card + `docs/`).

### Conversion image
- **Docker Hub:** `chankhavu/smolmo-olmocore-fast:1.4` (+ `:latest`). Public tooling image; no dataset/secrets baked in (dataset pulled at runtime with `HF_TOKEN`).

---

## 2. How to (re)generate the pre-tokenized dataset

The converter is **byte-identical** to the stock open-instruct `convert_sft_data_for_olmocore.py
--chat_template_name olmo_thinker`; it only removes single-threaded bottlenecks (see `docs/DESIGN.md`).

### Single machine
```bash
docker run --rm -e HF_TOKEN=hf_xxx -e PROCS=32 -e OUT_DATASET=chankhavu/<your-out-repo> \
  -v /data/out:/out chankhavu/smolmo-olmocore-fast:1.4
```

### Multi-node (split by parquet ids; one container per node, only PART_NUM differs)
```bash
# node k of K:
docker run --rm -e HF_TOKEN=hf_xxx -e PART_NUM=k -e NUM_PARTS=K -e PROCS=32 \
  -e OUT_DATASET=chankhavu/<your-out-repo> -v /out:/out chankhavu/smolmo-olmocore-fast:1.4
# after ALL nodes upload their parts to one olmocore/ folder:
docker run --rm -e MERGE_STATS=1 -e NUM_PARTS=K -e HF_TOKEN=hf_xxx -v /out:/out chankhavu/smolmo-olmocore-fast:1.4
```

### vast.ai (it erases the ENTRYPOINT — run the script yourself)
```bash
export HF_TOKEN=hf_xxx PART_NUM=k NUM_PARTS=K PROCS=32 WORK=/out OUT_DATASET=chankhavu/<your-out-repo>
bash /opt/convert/entrypoint.sh
```
Key env: `HF_TOKEN`, `PROCS` (≤ RAM/4, ≤ cgroup cap — see Gotchas), `WORK` (your big disk ≥ ~160 GB/node for K=4),
`DATASET` (source, default `chankhavu/smolmo-sft-v2-seqlen64k`), `OUT_DATASET` (upload target), `PART_NUM`/`NUM_PARTS`,
optional hard-coded `SHARD_START`/`SHARD_END`.

### OOM rescue (a node that finished tokenizing but died/hung at the stats step)
`image/rescue_part.py` rebuilds a node's final `.npy` from its `_tmp_part_K/_*.partial.bin` in 1 GB chunks
(low RAM), skipping the RAM-heavy aggregate step:
```bash
python3 /opt/convert/rescue_part.py /out/_tmp_part_006 /out/olmocore 6
```

---

## 3. What's in this repo (`/mnt/data/proof-redesign/olmocore-fast/`, git-tracked)

| path | what |
|---|---|
| `image/Dockerfile` | build context for `smolmo-olmocore-fast` (FROM `open-instruct-dataprep:0.1.0`) |
| `image/dataset_transformation.PATCHED.py` | open-instruct file, **patched** (3 hunks): `count_tokens` num_proc bounded by `PROCS`; parquet glob/list input; `import glob` |
| `image/numpy_dataset_conversion.PATCHED.py` | open-instruct file, **patched** (1 hunk): `SMOLMO_NO_SHUFFLE` env-gate (order-only, for deterministic multi-node) |
| `image/convert_node.py` | multi-node driver: split by parquet ids, node-prefixed parts, `--merge-stats`, completeness gate, stale-purge |
| `image/entrypoint.sh` | 3 modes — single / multi (`PART_NUM`+`NUM_PARTS`) / `MERGE_STATS`; download → convert → **stop for inspection** (no auto-upload) |
| `image/inspect_olmocore.py` | stats + decode/role-align an output dir |
| `image/rescue_part.py` | low-RAM `.bin` → `.npy` rescue (OOM recovery) |
| `patches/` | the open-instruct diffs: `*.patch` (unified diff), `*.ORIG.py`, `*.PATCHED.py` |
| `docs/DESIGN.md` | the redesign: baseline bottlenecks, the patches, file-subset sharding, fidelity argument, ordering decision |
| `docs/VERIFICATION.md` | the **final shipped-dataset verification** (all-green content scan + manual masking trace + audit summary) |
| `validate/` | the proofs (see §4) |

**The git log IS the audit trail** — each commit is one verified step:
```
git -C /mnt/data/proof-redesign/olmocore-fast log --oneline
```

---

## 4. Validation & audit records (`validate/`)

| file | proves |
|---|---|
| `validate/RESULTS.md` | **byte-identical** to stock olmo_thinker: Proof A (count_tokens patch, 300-row, matching sha256, 0 diffs) + Proof B (glob/list input on 3 real shards, 122 M tokens, identical sha256) |
| `validate/RESULTS_masking.md` | masking **role-alignment** correct for our data (system/user/environment masked, assistant trained, `<think>` opener masked) — open-instruct logic, unchanged |
| `validate/RESULTS_multinode.md` | C1 (multi-node split concatenates byte-identical to single-node) + C2 (no-shuffle = same document set, just reordered) |
| `validate/validate_patch.sh` | reproducible stock-vs-patched byte-diff harness |
| `validate/role_align.py` | decode + per-turn masked/trained check |
| `validate/proof_*/` | the stock-vs-patched stats JSONs from the proofs |
| `docs/VERIFICATION.md` | full content scan of the **shipped** 8-node dataset (totals exact, 0 corruption) + manual tool-use masking trace from the actual `.npy` |

Adversarial sub-agent councils (recorded in `git log` + `docs/`): Council #1 (patch fidelity), #2/#3
(multi-node coverage / naming / determinism), and a final 2-auditor pass on the shipped data — all GREEN.

---

## 5. Format / masking spec (reference)

Olmo-3 Thinking SFT format, **byte-identical to the stock `olmo_thinker` converter**:
- Reasoning opens with **`<think> `** (single space → clean `>`(masked) | ` Okay`(trained) loss boundary), closes `\n</think>\n\n` before the answer.
- Special tokens: `<|im_start|>`=100264, `<|im_end|>`=100265, `<|endoftext|>`=100257, `<|pad|>`=100277 (these are the ONLY special tokens; `<think>`, `<functions>`, `<function_calls>` are regular subword text).
- **Masked (label 0):** system, user, `environment` (tool outputs), and the `<|im_start|>assistant\n<think>` generation-prompt opener.
- **Trained (label 1):** assistant reasoning, the answer, `<function_calls>` tool calls, and the turn terminator.
- **Terminators (positional):** intermediate/tool-call assistant turns end `<|im_end|>`; the final assistant turn ends `<|endoftext|>` (both trained).
- Tools: declared in the (masked) system turn as `<functions>…</functions>`; the assistant emits `<function_calls>…</function_calls>` (trained); tool results come back as masked `environment` turns.

(Full details + token-level traces in `docs/VERIFICATION.md`; inference notes in the dataset's own `docs/INFERENCE_NOTES.md`.)
