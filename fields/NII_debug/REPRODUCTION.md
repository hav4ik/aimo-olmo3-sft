# Reproducing the NII run_01 `nvcc` crash locally

We reproduced the **exact** run_01 failure on our 2×RTX-3090 box using the **same shipped image**
(`olmo-sft-v2-allsm.sif`). Captured output: [`repro_output.log`](./repro_output.log).

## The idea
run_01 died because, during `torch.compile`, PyTorch ran `nvcc --version` and the `nvcc` it found
was **present but not executable** → `PermissionError` (which PyTorch does not catch). Our container
ships **no nvcc of its own** (verified: `command -v nvcc` → `NONE`), so to recreate NII's condition
we only have to put a **non-executable `nvcc` on `PATH`** — exactly what the NII host does by leaking
in a host nvcc that sits on a `noexec` / root-squashed mount.

## Step 0 — stage a non-executable nvcc (on the host)
```bash
mkdir -p /mnt/data/sif-test/fake
printf '#!/bin/sh\necho "fake nvcc 99.9"\n' > /mnt/data/sif-test/fake/nvcc
chmod 644 /mnt/data/sif-test/fake/nvcc      # readable, NO +x  == the host's nvcc that can't be exec'd
```
(`chmod 644` = "missing execute bit"; it produces the identical `EACCES` as a real `noexec` mount.
For a mount-faithful variant, `mount -t tmpfs -o noexec …`, drop a `chmod +x` nvcc in it, and bind
that instead — same error.)

## Quick check — the raw call (no GPU, ~1 s)
```bash
SIF=.../olmo-sft-v2-allsm.sif
singularity exec "$SIF" bash -lc '
  echo "baseline nvcc: $(command -v nvcc || echo NONE)"          # -> NONE (image has none)
  mkdir -p /tmp/fakebin
  printf "#!/bin/sh\necho fake\n" > /tmp/fakebin/nvcc; chmod 644 /tmp/fakebin/nvcc
  export PATH=/tmp/fakebin:$PATH
  python -c "import subprocess; subprocess.check_output([\"nvcc\",\"--version\"])"'
# -> PermissionError: [Errno 13] Permission denied: 'nvcc'
```

## Full path — torch.compile on the GPU (produces the run_01 traceback)
```bash
singularity exec --nv \
  --bind /mnt/data/sif-test/fake/nvcc:/usr/local/bin/nvcc \   # inject the broken nvcc onto PATH
  "$SIF" bash -lc '
    python - <<PY
import torch
@torch.compile(fullgraph=True)
def f(x): return (torch.sin(x)+torch.cos(x)).relu()
y=f(torch.randn(2048,2048,device="cuda")); torch.cuda.synchronize()
print("COMPILE OK", float(y.sum()))
PY'
# -> torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'
```
We bind at `/usr/local/bin/nvcc` (a dir that exists in the image and is on `PATH`). `/usr/local/cuda/bin`
would be even more NII-like, but that dir doesn't exist in our slim image, so its bind point is absent.

## ⚠️ The crash needs a COLD inductor cache
This is why it can look intermittent. `torch.compile` only shells out to `nvcc` while it **runs
codegen**. If the inductor/triton cache already holds the compiled graph (a warm cache from a prior
run with the same model/shapes), inductor loads it and **skips codegen entirely → never calls nvcc →
no crash**, even with the broken nvcc bound in. We hit exactly this: a first attempt with a warm
cache from earlier `test5` runs trained straight past the dry-run; after clearing the work dir
(cold cache) it crashed at the dry-run as expected. NII's run_01 crashed because a first run is
always cold. To force the crash deterministically regardless of leftover caches, add
`--env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` (or clear `…/work/node/<host>/{inductor,triton}` and any
`torchinductor_*`).

Full captured run (cold cache, real 1B training): [`repro_full_run.log`](./repro_full_run.log) —
crash at line ~1828 (`Starting forward/backward dry-run batch…` → `InductorError … 'nvcc'`).

## Reproduce inside the *real training run* (1B, your exact command)
Add **one** `--bind` line to the normal launch — it crashes at the dry-run batch, same as NII:
```bash
singularity run --nv --containall \
  --bind /mnt/data/sif-test/tmp:/tmp-wtf \
  --bind /mnt/data/sif-test/fake/nvcc:/usr/local/bin/nvcc \    # <-- the only addition
  --home "$PWD:/home/guest" --pwd /home/guest \
  olmo-sft-v2-allsm.sif \
  --experiment olmo_1b_bf16 --seq-len 8192 --run-suffix test5 \
  --max-steps 2000 --save-interval 1000 --ephemeral-interval 500 \
  --learning_rate 1e-12 --global-batch-tokens 16384
```

## Confirming the fix
Strip the CUDA dir from `PATH` so `nvcc` resolves to nothing → PyTorch's graceful "# nvcc not found"
path → compile + budget-mode AC keep working. Add to the command (keep the broken-nvcc bind to prove
it's dodged even when present):
```bash
  --env PATH=/usr/local/nvidia/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

## Verification that it matches run_01
The captured traceback in `repro_output.log` walks the identical frames as `run_01.log` (9687–9748):
`compile_fx.py:1241 codegen_and_compile → after_aot.save_graph_repro → generate_compiler_repro_string
→ debug_utils.py:265 _cuda_system_info_comment → subprocess.check_output(["nvcc","--version"]) →
PermissionError [Errno 13]`. Same image, same error, same stack.
