# 32B on H200 — fit test + pre-compiled binary (compile-cache) generation

Instructions for the agent running on a **4×H200** (or 8×H200) VastAI instance. One run does two jobs:
1. **Fit test** — confirm the 32B trains without OOM at the target per-rank shape.
2. **Pre-compiled binaries** — capture the warm Triton/inductor compile cache so the production run
   skips the fragile cold gcc/Triton compile (the `SIGABRT`-on-a-bad-node failure).

## Image + why these flags
Image: `chankhavu/olmo3-sft-v2.1:cu130-allsm` (it already contains the 32B SFT script + `olmo3_32b`
arch). We run at **seq 65536** (long-context, same as the 7B) with **16K tokens/GPU**. The image's
default 32B **batch/lr** are still the 7B's (1.05M / 5e-5) — fine for a fit test; override for a real run.

- `--seq-len 65536` — long-context (same as the 7B). This is also `run.sh`'s 32B default, so it matches
  what production will use.
- `--max-tokens-per-rank 16384` — the per-rank activation budget → `cp_degree=4` (65536/4 = 16384). On
  4 GPUs this gives mesh `dp_cp = shard(1) × cp(4) = 4`, so **FSDP shards the 32B across all 4 GPUs**
  (~¼ of params+optimizer each). This per-rank shape (16384 tokens) is **identical on 8×H200 with the
  same flags** (there cp=4, shard=2, dp_cp=8) — so the cache generated here is **valid for 8×H200
  production**, as long as production uses the same `--seq-len 65536 --max-tokens-per-rank 16384`.
- `--olmo-ac-budget 0.0` — recompute everything = **minimum memory**. Start here for the fit test; once
  it fits, raise it (0.5, 0.8) to find the throughput sweet spot, but 0.0 is the safe baseline.
- `--global-batch-tokens 65536` — memory is set by the *per-rank microbatch* (= seq), NOT the global
  batch, so use a tiny global batch here to keep `grad_accum=1` and the test fast. (Production uses the
  real batch; that only changes step count, not peak memory or the compiled kernels.)
- fp8 (`olmo_32b_fp8`) — the production precision (Hopper flash_3 path).

## Prereqs
- 4×H200 (or 8×) with `--gpus all`, docker, the image pulled.
- An HF token with read access to `allenai/Olmo-3.1-32B-Think` + the dataset.
- ~400 GB free host scratch (the 32B is ~64 GB download + distcp convert + cache).
- `--hostname cachehost` (deterministic cache path). The published image doesn't yet enforce the 4-var
  contract, but pass `WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400` anyway —
  harmless on the old image, required on the new one.

## Step 1 — fit test + cold compile (captures the cache)
Fresh workdir ⇒ cold cache ⇒ a complete compile is captured. Run the real entrypoint ~8 steps so the
forward, backward, loss, and optimizer kernels all compile.

```bash
mkdir -p /scratch/work /scratch/out
docker run --rm --gpus all --hostname cachehost \
  -e HF_TOKEN="$HF_TOKEN" -e WANDB_MODE=offline \
  -e WORLD_SIZE=1 -e GLOBAL_RANK=0 -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29400 \
  -v /scratch:/scratch \
  chankhavu/olmo3-sft-v2.1:cu130-allsm \
  --experiment        olmo_32b_fp8 \
  --seq-len           65536 \
  --max-tokens-per-rank 16384 \
  --olmo-ac-budget    0.0 \
  --global-batch-tokens 65536 \
  --workdir           /scratch/work \
  --output            /scratch/out \
  --max-steps         8 \
  --save-interval     100000 \
  --learning_rate     1e-12 \
  --run-suffix        prebuild32b \
  --no-remote-shell \
  2>&1 | tee /scratch/prebuild32b.log
```

**Watch for / report:**
- The GPU-memory line from olmo-core (`GPU capacity … X GiB allocated`) at peak — **report the peak
  allocated/reserved GiB per GPU** (this is the fit-test result).
- ✅ FIT = it reaches `step 1 … step 8` and exits 0 without `OutOfMemoryError`. If it **OOMs at budget
  0.0**, that's a real problem (32B doesn't fit on 4 GPUs even fully-recomputed) — capture the error and
  stop. If it fits, optionally re-run with `--olmo-ac-budget 0.5` then `0.8` and report the highest
  budget that still fits on 4×H200.

## Step 2 — capture the compile cache + manifest
```bash
SRC=/scratch/work/node/cachehost
mkdir -p /scratch/cache-out
cp -a "$SRC/triton"   /scratch/cache-out/triton
cp -a "$SRC/inductor" /scratch/cache-out/inductor
{
  echo "image_digest: $(docker inspect --format '{{index .RepoDigests 0}}' chankhavu/olmo3-sft-v2.1:cu130-allsm)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  docker run --rm --entrypoint python chankhavu/olmo3-sft-v2.1:cu130-allsm -c \
    'import torch,triton;print("torch",torch.__version__,"triton",triton.__version__,"cuda",torch.version.cuda)'
  echo "config: olmo_32b_fp8 seq=65536 max-tokens-per-rank=16384 (cp=4, per-rank 16384 tok) fp8"
  echo "triton_size: $(du -sh /scratch/cache-out/triton | cut -f1)"
  echo "inductor_size: $(du -sh /scratch/cache-out/inductor | cut -f1)"
} | tee /scratch/cache-out/MANIFEST.txt
```

## Step 3 — verify: complete + survives a broken compiler (the real test)
Seed the cache into a fresh, differently-named workdir and run with a **non-executable gcc + nvcc** to
prove no live compile is needed (i.e. a flaky cluster node would survive).

```bash
mkdir -p /scratch/verify/node/cachehost
cp -a /scratch/cache-out/triton   /scratch/verify/node/cachehost/triton
cp -a /scratch/cache-out/inductor /scratch/verify/node/cachehost/inductor
printf '#!/bin/sh\nexit 1\n' > /scratch/broken && chmod 000 /scratch/broken

docker run --rm --gpus all --hostname cachehost \
  -e HF_TOKEN="$HF_TOKEN" -e WANDB_MODE=offline \
  -e WORLD_SIZE=1 -e GLOBAL_RANK=0 -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29400 \
  -v /scratch:/scratch \
  -v /scratch/broken:/usr/bin/gcc:ro -v /scratch/broken:/opt/fields/bin/nvcc:ro \
  chankhavu/olmo3-sft-v2.1:cu130-allsm \
  --experiment olmo_32b_fp8 --seq-len 65536 --max-tokens-per-rank 16384 --olmo-ac-budget 0.0 \
  --global-batch-tokens 65536 --workdir /scratch/verify --output /scratch/out2 \
  --max-steps 3 --save-interval 100000 --learning_rate 1e-12 \
  --run-suffix verify32b --no-remote-shell --no-smoke-test \
  2>&1 | tee /scratch/verify32b.log
```
- ✅ PASS = reaches `step 1…3` despite the broken gcc/nvcc → cache is complete + relocatable.
- ❌ FAIL (gcc/nvcc/InductorError) = something still compiles live; bump Step 1 `--max-steps` to 16 and
  redo, and note which kernel is missing.

## Step 4 — package + report back
```bash
cd /scratch/cache-out
tar -I 'zstd -19 -T0' -cf /scratch/compile-cache-32b-fp8-sm90.tar.zst triton inductor MANIFEST.txt \
  || tar -czf /scratch/compile-cache-32b-fp8-sm90.tar.gz triton inductor MANIFEST.txt
sha256sum /scratch/compile-cache-32b-fp8-sm90.tar.* ; du -sh /scratch/compile-cache-32b-fp8-sm90.tar.*
grep -iE 'hf_|wandb|token|sk-' MANIFEST.txt prebuild32b.log verify32b.log   # must be empty
```
Hand back: the tarball, `MANIFEST.txt`, `prebuild32b.log`, `verify32b.log`, **and the two numbers that
matter — peak GiB/GPU at AC budget 0.0, and the highest AC budget that still fits on 4×H200.**

## Notes
- The cache is per-(model, seq, per-rank tokens, precision, arch, GPU). This one is for **32B fp8, seq
  65536, 16K tok/rank (cp=4)** — valid for 8×H200 production *only if production uses the same seq +
  max-tokens-per-rank*. If the production shape changes, regenerate (cheaply, same recipe) during the
  final 8×H200 pass.
- Don't bake any token into the tarball/logs (the grep above guards it).
- Compiling on the VastAI local disk is fine; on ABCI we'd point the per-node cache at `$PBS_LOCALDIR`.
