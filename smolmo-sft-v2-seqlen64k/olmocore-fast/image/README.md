# smolmo-olmocore-fast — single-node OLMo-core converter (download → convert → inspect)

Produces the OLMo-core pretokenized form of `chankhavu/smolmo-sft-v2-seqlen64k`
(`token_ids_part_*.npy` uint32 + `labels_mask_part_*.npy` bool + tokenizer + stats), **byte-identical**
to the stock open-instruct `convert_sft_data_for_olmocore.py --chat_template_name olmo_thinker`, but
much faster: it reads the 1024 shards **directly** (no re-write) and parallelizes the `count_tokens`
stats step. Tokenization + masking logic are the stock open-instruct code — **unchanged** (proven; see
`../validate/`).

The entrypoint does three things and then **STOPS** — it does NOT upload:
1. **Parallel download** the dataset from HuggingFace (all shards, `hf download --max-workers`).
2. **Convert** directly from the shards → `.npy` memmaps.
3. **Inspect**: prints stats + decodes a sample (role-alignment) so you can eyeball the data, then
   prints the exact `hf upload` command for **you** to run when satisfied.

## Run (single machine)
```bash
docker run --rm \
  -e HF_TOKEN=hf_xxx \           # read access to the (private) dataset
  -e PROCS=32 \                  # ~ memory-bandwidth sweet spot (more won't help; it's bandwidth-bound)
  -v /data/out:/out \            # writable, >= ~450 GB
  smolmo-olmocore-fast:1.0
# -> /data/out/olmocore/{token_ids_part_*.npy, labels_mask_part_*.npy, tokenizer/, dataset_statistics.json}
```
Env knobs: `HF_TOKEN`, `PROCS` (default nproc), `WORK` (default /out), `OUTPUT` (default $WORK/olmocore),
`DATASET` (default chankhavu/smolmo-sft-v2-seqlen64k), `DL_WORKERS` (download parallelism, default 16).

## Inspect, then upload yourself
The run ends by printing a summary and a decoded sample. Inspect more:
```bash
python /opt/convert/inspect_olmocore.py /data/out/olmocore --show 3   # decode + per-turn masked/trained
cat /data/out/olmocore/dataset_statistics.json
```
Expect: ~2,813,055 sequences, ~37.9B tokens, trainable ~0.85–0.95, max_seq 65536, 0 skipped; in the
decoded sample, system/user/`environment` MASKED and assistant TRAINED. When satisfied:
```bash
hf upload chankhavu/smolmo-sft-v2-seqlen64k /data/out/olmocore olmocore --repo-type dataset   # WRITE token
```

## Hardware (this is MEMORY-BANDWIDTH bound)
- Speed comes from aggregate memory bandwidth, not cores — tokenization saturates ~8–16 cores/socket.
  Prefer DDR5 / multi-channel / EPYC; a desktop dual-channel box is the slow floor.
- `PROCS ~32` is plenty; higher just eats RAM (~1.5–2 GB/proc).
- **Disk: mount >= ~450 GB at /out** (HF download ~35 GB + Arrow cache + ~190 GB output).

## Fidelity
Tokenization/masking are stock open-instruct (`olmo_thinker`), untouched. The only changes are in
`dataset_transformation.py`: `count_tokens` gains `num_proc` (stats only), and the parquet input
dispatch accepts a glob/list (so shards feed directly). Proven byte-identical (`token_ids` +
`labels_mask` matching SHA-256, 0 diffs) — see `../validate/RESULTS.md` and `../patches/`.

> Multi-node: not in this image (single-node only). The planned multi-node path uses deterministic
> file-subset ordering (no seed reliance) — see `../docs/DESIGN.md`.
