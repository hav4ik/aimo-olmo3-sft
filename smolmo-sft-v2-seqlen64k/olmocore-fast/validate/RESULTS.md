# Validation results — byte-identical proofs

Both proofs use the **stock** open-instruct converter (image `open-instruct-dataprep:0.1.0`,
`convert_sft_data_for_olmocore.py --chat_template_name olmo_thinker --max_seq_length 65536`) as the
ground truth, and diff the resulting `.npy` byte streams against the **patched** converter. The patch
changes only `count_tokens` parallelism and parquet-input dispatch (see patches/dataset_transformation.patch).

## Proof A — `count_tokens` num_proc patch (same single input)
Input: 300-row sample.parquet. Stock vs patched (count_tokens map now `num_proc`).
- `token_ids`   : 4,170,765 tokens — **sha256 identical**, 0 diffs
- `labels_mask` : 4,170,765       — **sha256 identical**, 0 diffs
- stats: 300 seq / 4,170,765 tok / 3,750,761 trainable / 0 skipped (both runs match)

## Proof B — direct glob/list input (real shards) — DECISIVE
Same 3 real dataset shards (`train-00000/00001/00002-of-01024`, 9,000 rows), two paths:
- **stock**: shards concatenated (in order) into one `ref3.parquet`, fed to the stock converter
- **patched**: the 3 shard files passed directly as a comma-list to `--dataset_mixer_list` (glob/list dispatch)

Diff of concatenated output streams:
| stream | tokens | stock sha256[:16] | patched sha256[:16] | identical | diffs |
|---|---|---|---|---|---|
| token_ids (uint32)  | 122,003,488 | cd2dd8159be7cb5b | cd2dd8159be7cb5b | YES | 0 |
| labels_mask (bool)  | 122,003,488 | 3e2fedbcbe55a6a6 | 3e2fedbcbe55a6a6 | YES | 0 |

Stats match exactly: 9,000 seq / 122,003,488 tok / 109,791,917 trainable / 0 skipped (both runs).

## Conclusion
Feeding the 1024 shards directly (glob/list) and parallelizing `count_tokens` produces **byte-for-byte
identical** token_ids and labels_mask vs the stock olmo_thinker path. Tokenization + masking are
untouched; only plumbing (file read + stats counting) changed.

Reproduce: `PATCHED_DT=<patched.py> validate/validate_patch.sh <input> <workdir>`.
