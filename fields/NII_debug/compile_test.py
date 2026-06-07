#!/usr/bin/env python3
"""Minimal reproducer for the NII run_01 crash: torch.compile shelling out to a
non-executable `nvcc` and dying with PermissionError [Errno 13].

Run it three ways to confirm the theory + the fix:
  1) plain                        -> COMPILE OK            (nvcc executable, baseline)
  2) with a non-exec nvcc on PATH -> PermissionError 13    (reproduces NII run_01)
  3) + TORCH_COMPILE_DISABLE=1    -> COMPILE OK (skipped)  (confirms the escape hatch)

See the sibling shell recipes for how to stage the non-exec nvcc.
"""
import os, shutil, subprocess, torch

print(f"torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
print(f"nvcc resolved by shutil.which -> {shutil.which('nvcc')}")
# What torch actually does (list form, no shell, execvp semantics):
try:
    subprocess.check_output(["nvcc", "--version"])
    print("direct `nvcc --version` -> ran OK")
except Exception as e:  # noqa: BLE001
    print(f"direct `nvcc --version` -> {type(e).__name__}: {e}")

@torch.compile(backend="inductor", fullgraph=True)
def f(x):
    return (torch.sin(x) + torch.cos(x)).relu()

x = torch.randn(2048, 2048, device="cuda")
try:
    y = f(x); torch.cuda.synchronize()
    print(f"COMPILE OK  sum={float(y.sum()):.3f}  (TORCH_COMPILE_DISABLE={os.environ.get('TORCH_COMPILE_DISABLE')})")
except Exception as e:  # noqa: BLE001
    print(f"COMPILE FAILED: {type(e).__name__}: {e}")
    raise
