# SFT container crash on NII — diagnosis & local reproduction

## Cause
The job crashed before training step 0 with `PermissionError: [Errno 13] Permission denied: 'nvcc'`,
on every GPU rank, during the compile that PyTorch runs at the first (dry-run) batch.

Our container intentionally ships **no `nvcc`** — it needs none at runtime (the CUDA *runtime*
libraries come from bundled pip packages, FlashAttention is a prebuilt binary wheel, and Triton
compiles its kernels with its own bundled `ptxas`, not `nvcc`). Because the image has no `nvcc` of its
own, the process resolves `nvcc` from the **host** (a host CUDA directory that is on `PATH` and visible
inside the container). On the NII node that host `nvcc` is **present but not executable for the job** —
the signature (`EACCES`, errno 13, raised while *launching* the binary) is exactly what a **`noexec`
mount** or a **root-squashed NFS** file produces. This is standard cluster security; nothing on the
NII side is misconfigured.

The reason it is *fatal* rather than harmless: PyTorch's helper calls `nvcc --version` only to embed a
version string in a debug comment, and it **catches a *missing* nvcc** (`FileNotFoundError` →
prints "# nvcc not found" and continues) — but it does **not** catch a *non-executable* one, so the
`PermissionError` propagates and kills every rank. That asymmetry is why the same image runs fine on
our workstation (no host `nvcc` in the way → "not found" → ignored) but fails on NII (a
non-executable host `nvcc` is present). For completeness: this is unrelated to disk space — a full
disk raises a different error (`ENOSPC`, errno 28) from *write* calls and cannot make launching a
binary fail; there are no out-of-space messages in the run log.

## The error
The relevant frames (identical across all ranks):
```
olmo_core.train.trainer   Starting forward/backward dry-run batch...
torch/_inductor/compile_fx.py        codegen_and_compile
torch/_dynamo/repro/after_aot.py     save_graph_repro -> generate_compiler_repro_string
torch/_dynamo/debug_utils.py:265     cuda_version_out = subprocess.check_output(["nvcc", "--version"])
torch._inductor.exc.InductorError:   PermissionError: [Errno 13] Permission denied: 'nvcc'
```
Note the call is in inductor's *repro/system-info* path (a version probe), **not** real kernel
compilation — so the container does not actually need `nvcc` to work; it only needs the probe not to
crash.

## We reproduced it exactly (proof)
On our 2×GPU workstation, using the **same shipped image** (`olmo-sft-v2-allsm.sif`), we recreated the
NII condition by binding a deliberately **non-executable** `nvcc` onto the container's `PATH`
(`--bind <fake nvcc>:/usr/local/bin/nvcc`, the file having no execute permission) and launched the
normal 1B training command. It crashed at the dry-run batch with a **frame-for-frame identical**
traceback. The full run, from launch through download, base-model conversion, model build,
`torch.compile`, and the crash, is captured in `repro_full_run.log`.

Two independent cross-checks confirm it is the *same* failure, not a look-alike:

1. **Per-rank signature matches.** Every crashing GPU process emits the same three lines —
   `ERROR Training failed due to…`, `CRITICAL Uncaught InductorError…`, and the final
   `…PermissionError [Errno 13] Permission denied: 'nvcc'`. The NII run logged **24** such lines =
   **8 GPUs × 3**; our 2-GPU reproduction logged **6** = **2 GPUs × 3**. Same crash, once per rank;
   the only difference is the GPU count.

2. **It requires a cold compile cache.** `torch.compile` invokes `nvcc` only while it actually runs
   code generation. If the inductor/triton cache is already warm (a previous run compiled the same
   model and shapes), code generation is skipped and the bad `nvcc` is never touched. So the crash
   hits the **first, cold run** of a fresh job — exactly the NII case — and can appear to "go away"
   once a cache exists. We observed this directly: our first local attempt had a warm cache from
   earlier test runs and trained straight past the dry-run; after clearing the cache (cold compile)
   it crashed as expected. Practically, on NII this means the failure is not flaky — it will recur on
   every fresh run until either `nvcc` is fixed or the cache is pre-warmed.

## Please confirm (≈5 s, in the same container/session that failed)
This verifies our hypothesis — that `nvcc` resolves to a **host path** sitting on a **`noexec`** (or
root-squashed) mount:
```bash
which -a nvcc                                          # which nvcc is picked up, and from where?
N=$(readlink -f "$(which nvcc)"); ls -l "$N"; stat -c '%A %U:%G' "$N"; id
findmnt -T "$N" -o TARGET,SOURCE,FSTYPE,OPTIONS        # is 'noexec' in OPTIONS? is SOURCE a host fs?
nvcc --version                                         # should reproduce the Permission denied
```
How to read it: a `nvcc` path **outside the container image** confirms it's the host's; `noexec` in
the mount OPTIONS confirms the mount-level cause; alternatively an exec-allowed mount but a mode
without `o+x` / an owner that isn't the job user (visible in `stat` + `id`) points to root-squash
permissions. Any of these matches our diagnosis.

## Fix (entirely on our side — no NII change required)
- **Immediate unblock, no new image needed.** Add an env override to the `singularity run` command
  that removes the CUDA directory from `PATH`, so `nvcc` resolves to nothing and PyTorch takes its
  graceful "# nvcc not found" path. This preserves full functionality (compilation and
  activation-checkpointing budget mode keep working):
  ```
  --env PATH=/usr/local/nvidia/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  ```
- **Permanent (SHIPPED in v2.1, verified).** The image now ships its own `nvcc` at `/opt/fields/bin/nvcc`
  (first on `PATH`), so the container never resolves (or depends on) the host's — regardless of `--nv`
  or host environment. Verified end-to-end: with a non-executable nvcc bound at the NII vector
  `/usr/local/cuda/bin/nvcc` the container now passes the dry-run compile and trains (see
  `full_container_test.log`). Use image `olmo-sft-v2.1-allsm.sif` / `chankhavu/olmo3-sft-v2.1:cu130-allsm`.

**Attachment:** `repro_full_run.log` — the complete local run, launch to crash, on the shipped image.
