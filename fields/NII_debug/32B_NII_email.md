# 32B SFT run — note for NII (email-ready)

## Disk volume requirement

### One-liner / ask
The 32B SFT run needs a per-node shared volume (the `--workdir` / output mount, on the shared Lustre)
of **~1.3–1.5 TB** for comfortable operation. **If 1 TB is a hard cap, we have a validated fallback
configuration that peaks at ~0.9 TB** — details below. The 7B run is unaffected (its checkpoints are
~½ the size).

### Where the disk goes
One shared volume holds all of these *concurrently* during a run:

| item | size | notes |
|---|---|---|
| tokenized dataset | ~200 GB | the SFT data, read locally by every rank |
| base model — HF download | ~64 GB | `allenai/Olmo-3.1-32B-Think` safetensors (reclaimable after convert) |
| base model — distcp conversion | ~64 GB | the starting checkpoint training loads |
| HF-format convert copy | ~64 GB | created by our uploader to ship each checkpoint to HuggingFace |
| **training checkpoint (each)** | **~251 GB** | distcp = model **+** optimizer state; the dominant term |

A single 32B distcp checkpoint is ~251 GB, and olmo-core **writes the new checkpoint before pruning the
old one** (atomic write-then-rename, then prune), so a save *transiently* holds one extra checkpoint.

### Option A — preferred (~1.3–1.5 TB)
`keep_last=1` → **1 persistent + 1 rotating ephemeral** checkpoint on disk: gives a rollback checkpoint
plus 250-step crash-resume granularity.
- steady-state: ~830–895 GB
- worst-case spike (a persistent save in flight, overlapping the ephemeral + the HF convert copy):
  **~1145 GB**

### Option B — 1 TB floor (ephemeral-only)
Set the persistent save interval beyond the total step count → **no intermediate persistent checkpoints**;
rely on one rotating ephemeral for crash-resume. olmo-core still writes a **guaranteed final checkpoint at
the last step** (independent of the interval), so the deliverable is unaffected.
- steady-state: ~580–645 GB
- worst-case spike (an ephemeral save in flight): **~895 GB** → fits 1 TB
- trade-off: only one local resume copy (no rollback if that single checkpoint is ever corrupted). Our
  uploader still ships inference/soup checkpoint snapshots to HuggingFace throughout the run, so the
  checkpoint *history* is preserved off-volume regardless.

### Notes / caveats
- The **~251 GB** figure is a prior measurement; with bf16 optimizer state (our default,
  `OLMO_OPTIM_DTYPE=bf16`) the real distcp may be meaningfully smaller — we will confirm the exact size on
  the 8×H200 validation run (watch `df` during the first save). If it is smaller, Option A's headroom on
  1 TB improves accordingly.
- The **base-model artifacts (~128 GB)** currently persist for the whole run; the HF-download half
  (~64 GB) is reclaimable immediately after the distcp conversion (a re-run skips convert via a
  completion sentinel, so it is never needed again).
- ABCI/NII group storage is petabyte-scale, so a 1 TB limit is most likely a default quota or node-local
  scratch size rather than a physical wall — a bump to ~1.5 TB on the shared area is the simplest path.

### What we need from NII
Either: a **~1.3–1.5 TB** shared volume for the 32B run (preferred), **or** confirmation that **1 TB** is
the cap — in which case we ship with the ephemeral-only checkpoint config (Option B), which we have
validated fits.
</content>
