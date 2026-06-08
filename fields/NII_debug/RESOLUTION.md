# RESOLUTION — container crashed on the host's non-executable `nvcc`

**TL;DR:** Our SFT container shipped **no `nvcc`** of its own. On the NII node a **host** `nvcc` (on a
`noexec` mount — normal cluster security) got picked up via `PATH`, and `torch.compile` crashed every
rank before step 0. Fixed in **v2.1** by shipping our own `nvcc` first on `PATH`. NII run is live + fast. ✅

## What happened
```
torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'
```
During the first (dry-run) compile, PyTorch shells out to `nvcc --version`. The subtle part:

> PyTorch **catches a _missing_ nvcc** (`FileNotFoundError` → "# nvcc not found", harmless) but **not a
> _non-executable_ one** (`PermissionError` → fatal).

So a `nvcc` that's *present but not executable* (noexec mount / missing `+x` / root-squashed NFS) is the
one fatal case. It only triggers on a **cold** Triton/inductor cache, so it nails the first run of a
fresh node and can look intermittent.

## The fix (v2.1)
- **Ship our own `nvcc`** (CUDA-13.0 pip wheels) at `/opt/fields/bin/nvcc`, **first on `PATH`** — so the
  container never resolves (or depends on) a host `nvcc`. Image: `chankhavu/olmo3-sft-v2.1:cu130-allsm`.
- Plus hardening so the *whole* toolchain (gcc/g++/as/ld) resolves in-image first, and the compilers are
  pinned (`CC`/`CXX`/`-ccbin`). The only host dependency left is the GPU driver via `--nv` (correct).

## Lesson (applies to any GPU container)
A container should be **self-contained for its build/compile toolchain**; never rely on a host binary
resolved through `PATH`. The host's `noexec` mounts are *security*, not a bug — fix it on the image side.

## Minimal test — does a container survive a non-executable host `nvcc`?
Run **inside the container, on a GPU node**. It plants a present-but-non-executable `nvcc` on `PATH`
(the exact NII condition) and runs a cold `torch.compile`:
```bash
D=$(mktemp -d); printf '#!/bin/sh\necho fake\n' > "$D/nvcc"; chmod 0644 "$D/nvcc"   # present, NOT +x
PATH="$PATH:$D" TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python - <<'PY'
import torch
torch.compile(lambda x: (x.sin()+x.cos()).relu(), fullgraph=True)(
    torch.randn(512, 512, device="cuda")).sum().item()
print("PASS — container has its own working nvcc (or handles its absence). Robust.")
PY
# Robust image -> "PASS". Vulnerable image -> InductorError: PermissionError [Errno 13] 'nvcc'.
```
Why it works: if the image ships its own `nvcc` earlier on `PATH`, it wins and you get `PASS`; if it
ships none, the planted non-exec one is the only `nvcc` → the crash reproduces.

Even more minimal — the mechanism itself (no GPU needed):
```python
import subprocess
try: subprocess.check_output(["nvcc", "--version"])
except FileNotFoundError: print("absent -> torch handles it (safe)")
except PermissionError as e: print(f"present-but-not-executable -> FATAL in torch.compile: {e}")
```

## More
Full repro/audit harness + the email we sent NII: `fields/NII_debug/` — `nvcc_selfcheck.sh`
(`diagnose`/`test` an image or `.sif`), `NII_REPORT.md`, `REPRODUCTION.md`, `HANDOFF.md`.
