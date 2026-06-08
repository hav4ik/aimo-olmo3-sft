# Build a warm Triton/inductor compile cache (H200) — instructions for the agent

**Goal:** produce a portable, pre-compiled Triton + inductor cache on an **H200 (sm_90)** so that on the
NII cluster the first (cold) `torch.compile` is a **cache hit** instead of a live gcc/Triton compile.
That compile is what crashed `hnode565` with `InductorError: CalledProcessError ... gcc ... SIGABRT`
(a flaky/locked-down node), while `hnode532` compiled fine and is training. A baked warm cache makes the
run survive **any** node, good or bad.

You (the H200 agent) only do **Part A** (generate + verify + hand back the cache tarball). **Part B**
(bake it into the image + seed it per-node at startup) is done back in the main repo — it's described at
the end only so you know what the deliverable must contain.

---

## Why the cache has to match (these are literally the cache keys)

A Triton/inductor cache entry is keyed by a hash of: **GPU arch**, **Triton/PyTorch/inductor/ptxas/gcc
versions**, the **compiled graph** (which comes from tensor shapes), and inductor config. If any of these
differ between your warm-up and the NII run, that entry **misses** and NII recompiles it live. So:

| Must match | Why | How you guarantee it |
|---|---|---|
| **Same image** `chankhavu/olmo3-sft-v2.1:cu130-allsm` | fixes torch/Triton/inductor/ptxas/gcc versions **and** the absolute paths inside the container | run the warm-up **in this exact image** |
| **H200 / sm_90** | kernels are arch-specific; a 3090/sm_86 cache is useless on H200 | run on an actual H200 |
| **Same launch shape** (`--experiment`, `--seq-len`, `--max-tokens-per-rank`, `--olmo-ac-budget`) | the compiled graph keys on per-rank tensor shapes | pass the **exact production flags** below |
| **Same model** (7B vs 32B) | different arch → different kernels | one cache per model you intend to run |

**Relocatability (good news):** the cache is keyed by content hash, not by its directory path, so a cache
generated under `/scratch/...` will hit when NII reads it from `<workdir>/node/<host>/...`. The verify
step below proves this by moving the cache to a different path.

**Topology-independent (good news):** the compiled graph is the **per-rank** forward/backward, which does
not depend on how many GPUs you use — only on `cp_degree` and the per-rank token count. At the 7B-fp8
shape `cp=2`, so you need **≥2 H200** (4×H200 is ideal and matches a common VastAI shape). A 4×H200 cache
is identical to an 8×H200 cache.

---

## Prerequisites

- An H200 box (≥2 GPUs for the 7B-fp8 `cp=2` shape; 4× recommended).
- Docker with `--gpus all`.
- `chankhavu/olmo3-sft-v2.1:cu130-allsm` pulled (`docker pull` it; record the digest).
- An **HF token** with read access to `allenai/Olmo-3-7B-Think` and the SFT dataset (the warm-up really
  downloads the model + a little data — that's required to compile the real graph).
- A host scratch dir with ~60 GB free (model + dataset + cache). Call it `/scratch` below.

> We use `--hostname cachehost` on every `docker run` so the per-node cache lands in a **deterministic**
> `node/cachehost/...` path (instead of a random container ID). That makes capture + verify trivial.

---

## Part A — generate, verify, hand back

### Step 1 — cold warm-up run (compile everything once)

Fresh workdir ⇒ cold cache ⇒ a complete compile is captured. Run the **real entrypoint** for a handful of
steps (don't hand-write a compile harness — the real run compiles exactly what production compiles:
forward, backward, loss, optimizer).

```bash
mkdir -p /scratch/work /scratch/out
docker run --rm --gpus all --hostname cachehost \
  -e HF_TOKEN="$HF_TOKEN" \
  -e WANDB_MODE=offline \
  -v /scratch:/scratch \
  chankhavu/olmo3-sft-v2.1:cu130-allsm \
  --experiment        olmo_7b_fp8 \
  --max-tokens-per-rank 32768 \
  --olmo-ac-budget    0.8 \
  --workdir           /scratch/work \
  --output            /scratch/out \
  --max-steps         12 \
  --save-interval     100000 \
  --learning_rate     1e-12 \
  --run-suffix        cachewarm \
  --no-remote-shell \
  2>&1 | tee /scratch/warmup.log
```

Notes:
- `--seq-len` is omitted ⇒ recipe default **65536**; with `--max-tokens-per-rank 32768` that's **cp=2,
  32768 tokens/rank** — the production shape. Do **not** change these two; they define the graph.
- `--olmo-ac-budget 0.8` matches production. (AC budget changes which subgraphs are compiled, so it's a
  cache key too — keep it at the production value.)
- `--max-steps 12` is enough to get past the dry-run compile **and** a few real optimizer steps (the
  optimizer/loss kernels compile on the first real step). `--learning_rate 1e-12` + huge `--save-interval`
  ⇒ it doesn't meaningfully train and writes no checkpoint.
- Let it download the model/dataset normally. Expect ~15–40 min total on H200 (download dominates).
- ✅ Success = it prints training step logs (`step 1 … step 12`) and exits 0. If it dies at the dry-run
  with a gcc/nvcc error, the **box itself** has the same toolchain problem — try a different H200.

### Step 2 — capture the two cache dirs + a manifest

```bash
SRC=/scratch/work/node/cachehost
ls -la "$SRC"            # expect: triton/  inductor/  (plus home/ tmp/)
mkdir -p /scratch/cache-out
cp -a "$SRC/triton"   /scratch/cache-out/triton
cp -a "$SRC/inductor" /scratch/cache-out/inductor
# belt-and-suspenders: if inductor wrote anything under HOME, grab it too (usually empty)
[ -d "$SRC/home/.triton" ] && cp -a "$SRC/home/.triton" /scratch/cache-out/home_triton || true

# manifest — record EXACTLY what this cache is keyed to
{
  echo "image_digest: $(docker inspect --format '{{index .RepoDigests 0}}' chankhavu/olmo3-sft-v2.1:cu130-allsm)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  docker run --rm --entrypoint python chankhavu/olmo3-sft-v2.1:cu130-allsm -c \
    'import torch,triton;print("torch:",torch.__version__);print("triton:",triton.__version__);print("cuda:",torch.version.cuda)'
  echo "flags: --experiment olmo_7b_fp8 --seq-len 65536 --max-tokens-per-rank 32768 --olmo-ac-budget 0.8 (cp=2)"
  echo "triton_size: $(du -sh /scratch/cache-out/triton | cut -f1)"
  echo "inductor_size: $(du -sh /scratch/cache-out/inductor | cut -f1)"
  echo "file_count: $(find /scratch/cache-out -type f | wc -l)"
} | tee /scratch/cache-out/MANIFEST.txt
```

### Step 3 — verify the cache is COMPLETE and RELOCATABLE (the important test)

This is the proof that the cache makes a **bad node** survivable. Seed the captured cache into a **fresh,
differently-named workdir**, then run with a **deliberately broken gcc + nvcc**. If training still reaches
the steps, then no live compile was needed ⇒ a flaky-compiler node would have survived.

```bash
# seed a fresh workdir from the captured cache (different path ⇒ proves relocatability)
mkdir -p /scratch/verify/node/cachehost
cp -a /scratch/cache-out/triton   /scratch/verify/node/cachehost/triton
cp -a /scratch/cache-out/inductor /scratch/verify/node/cachehost/inductor

# a non-executable "compiler" to simulate the bad node (chmod 000)
printf '#!/bin/sh\nexit 1\n' > /scratch/broken && chmod 000 /scratch/broken

docker run --rm --gpus all --hostname cachehost \
  -e HF_TOKEN="$HF_TOKEN" -e WANDB_MODE=offline \
  -v /scratch:/scratch \
  -v /scratch/broken:/usr/bin/gcc:ro \
  -v /scratch/broken:/opt/fields/bin/nvcc:ro \
  chankhavu/olmo3-sft-v2.1:cu130-allsm \
  --experiment olmo_7b_fp8 --max-tokens-per-rank 32768 --olmo-ac-budget 0.8 \
  --workdir /scratch/verify --output /scratch/out2 \
  --max-steps 3 --save-interval 100000 --learning_rate 1e-12 \
  --run-suffix cacheverify --no-remote-shell --no-smoke-test \
  2>&1 | tee /scratch/verify.log
```

- ✅ **PASS** = it reaches `step 1 … step 3` and exits 0 **despite** the broken gcc/nvcc. The warm cache is
  complete; a bad node would survive. (The model still downloads — that's fine; we're only testing compile.)
- ❌ **FAIL** = it dies with a gcc/nvcc/`CalledProcessError`/`InductorError`. That means something still
  compiles live and isn't in the cache — capture more (raise `--max-steps` in Step 1 to 20 and redo), or
  note which kernel is missing in the log so we can investigate.

### Step 4 — package and hand back

```bash
cd /scratch/cache-out
tar -I 'zstd -19 -T0' -cf /scratch/compile-cache-7b-fp8-sm90.tar.zst triton inductor MANIFEST.txt
#   (no zstd? use: tar -czf /scratch/compile-cache-7b-fp8-sm90.tar.gz triton inductor MANIFEST.txt)
sha256sum /scratch/compile-cache-7b-fp8-sm90.tar.zst
du -sh    /scratch/compile-cache-7b-fp8-sm90.tar.zst
```

Hand back: **the tarball**, `MANIFEST.txt`, `warmup.log`, and `verify.log`. Report the tarball size, the
sha256, and whether Step 3 PASSed. Do **not** put any HF/W&B token into the tarball, the manifest, or the
logs (grep them: `grep -iE 'hf_|wandb|token|sk-' MANIFEST.txt warmup.log verify.log` should be empty).

---

## Gotchas

- **cp=2 needs ≥2 GPUs.** On a single H200 the run can't form the `cp=2` mesh. Use 2/4/8 H200.
- **Device-name autotune misses are OK.** A few inductor autotune entries key on the exact device string
  (e.g. `H200` vs `H200 NVL`); if NII's SKU differs, those re-autotune — but the **gcc `cuda_utils` compile
  that caused the SIGABRT is arch-keyed, not device-name-keyed, so it still hits.** The dangerous compile
  is skipped either way. (If NII's `nvidia-smi` name differs from your box's, note it in the report.)
- **One cache per (model, precision, shape).** This produces the **7B-fp8** cache. For the **32B**, repeat
  Steps 1–4 with `--experiment olmo_32b_fp8` and the **exact 32B production flags** (see `SCALEUP_32B.md`
  in the repo — it needs enough H200s to fit the 32B), and name the tarball `…-32b-fp8-sm90.tar.zst`.
- **Image version lock.** This cache is valid only for `olmo3-sft-v2.1:cu130-allsm`'s torch/Triton/cuda.
  If we later bump any of those, the cache must be regenerated. (The Part-B seeding change is `train.py`
  only — no version bump — so it does **not** invalidate this cache.)

---

## Part B — what happens to the cache back in the repo (FYI, not your job)

The cache can't just be dropped at `node/<hostname>/...` because the hostname is unknown at build time and
differs per node. So in the main repo we will:
1. **Bake** the cache into the image at a read-only path, e.g. `/opt/fields/compile-cache/7b-fp8/{triton,inductor}`.
2. **Seed it per-node at startup** in `fields/train.py` (right where it sets `TRITON_CACHE_DIR` /
   `TORCHINDUCTOR_CACHE_DIR`, ~line 617): if the per-node `node/<host>/{triton,inductor}` dirs are empty,
   copy the baked cache into them. Every node — including a flaky one — then starts **warm** and never runs
   the fragile gcc/Triton compile.

That's why your deliverable just needs to be a clean, verified `{triton, inductor}` pair with a manifest.
