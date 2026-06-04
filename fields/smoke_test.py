#!/usr/bin/env python3
"""Fields Model — preflight smoke test for the Singularity container.

Run this FIRST inside the built .sif to confirm the environment is sound before a real training run
(VastAI can't run Singularity, so this is how you validate the .sif locally)::

    singularity exec --nv <name>_train_<date>.sif python /app/smoke_test.py --workdir /scratch --output /results

It checks, and prints a PASS/WARN/FAIL report:
  * GPUs visible to the container (nvidia-smi) + torch CUDA + compute capability (sm_90 expected on H200)
  * credentials: `hf auth whoami` and `wandb login --verify`
  * the core files exist (train.py, upload.py, run.sh, SFT script, the HF converters)
  * the workdir / output dirs are writable, with enough free disk for the run
  * the OLMo-core stack imports and works (olmo_core, flash-attn, transformer_engine + the libnvrtc
    linker fix, ring-flash-attn, the to_hf converter), and FP8 (torch._scaled_mm) is available
  * HuggingFace hub reachability for the base model download

Exit code is non-zero iff a CRITICAL check fails (the container can't train). WARNs do not fail the run.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

try:
    from secrets_loader import load_secrets
except ImportError:
    def load_secrets(path: Optional[str] = None) -> Optional[str]:
        return None

OLMO_CORE_ROOT = Path(os.environ.get("OLMO_CORE_ROOT", "/workspace/OLMo-core"))
APP = Path(os.environ.get("FIELDS_APP", "/app"))
CODE = Path(os.environ.get("FIELDS_CODE_ROOT", "/app/code"))
CONVERTERS = OLMO_CORE_ROOT / "src" / "examples" / "huggingface"
MIN_FREE_GB = float(os.environ.get("FIELDS_MIN_FREE_GB", "150"))  # 7B: base distcp + dataset + checkpoints

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, msg: str = "") -> None:
    _results.append((name, status, msg))
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
    print(f"  [{icon}] {name:32s} {status:4s} {msg}")


def guard(name: str, fn: Callable[[], tuple[str, str]], critical: bool = True) -> None:
    """Run a check; turn any exception into FAIL (critical) or WARN (non-critical)."""
    try:
        status, msg = fn()
    except Exception as exc:  # noqa: BLE001
        status, msg = (FAIL if critical else WARN), f"{type(exc).__name__}: {exc}"
    record(name, status, msg)


def sh(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


# ---- checks -------------------------------------------------------------------------------------
def c_python() -> tuple[str, str]:
    v = sys.version_info
    return PASS, f"python {v.major}.{v.minor}.{v.micro}"


def c_nvidia_smi() -> tuple[str, str]:
    rc, out = sh(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"])
    if rc != 0:
        return FAIL, f"nvidia-smi failed (Singularity --nv missing?): {out[:160]}"
    gpus = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return (PASS if gpus else FAIL), f"{len(gpus)} GPU(s): {gpus[0] if gpus else 'none'}"


def c_torch_cuda() -> tuple[str, str]:
    import torch
    if not torch.cuda.is_available():
        return FAIL, "torch.cuda.is_available() == False"
    n = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0)
    note = "" if cap[0] >= 9 else "  (expected sm_90+ on H200)"
    return PASS, f"torch {torch.__version__} | {n} GPU | sm_{cap[0]}{cap[1]}{note}"


def c_fp8() -> tuple[str, str]:
    import torch
    if not hasattr(torch, "_scaled_mm"):
        return WARN, "torch._scaled_mm absent (FP8 path unavailable)"
    cap = torch.cuda.get_device_capability(0)
    return PASS, f"torch._scaled_mm present; FP8-capable arch sm_{cap[0]}{cap[1]}"


def c_hf_whoami() -> tuple[str, str]:
    if not os.environ.get("HF_TOKEN"):
        return WARN, "HF_TOKEN not set (no SECRETS.json / env) — downloads of private repos will fail"
    rc, out = sh(["hf", "auth", "whoami"])
    if rc != 0:
        return WARN, f"hf auth whoami failed: {out[:160]}"
    return PASS, f"hf user: {out.splitlines()[0][:80]}"


def c_wandb() -> tuple[str, str]:
    if not os.environ.get("WANDB_API_KEY"):
        return WARN, "WANDB_API_KEY not set — training will run W&B offline"
    rc, out = sh(["wandb", "login", "--verify"])
    return (PASS if rc == 0 else WARN), out.splitlines()[-1][:120] if out else f"rc={rc}"


def c_core_files() -> tuple[str, str]:
    required = [
        APP / "train.py", APP / "upload.py", APP / "smoke_test.py", APP / "secrets_loader.py",
        CODE / "olmocore" / "run.sh",
        CODE / "olmocore" / "sft_scripts" / "Olmo-3-7B-SFT-local.py",
        CONVERTERS / "convert_checkpoint_from_hf.py",
        CONVERTERS / "convert_checkpoint_to_hf.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return FAIL, f"missing: {', '.join(missing)}"
    return PASS, f"{len(required)} core files present"


def c_writable() -> tuple[str, str]:
    args = parse_args()
    notes = []
    for label, d in (("workdir", Path(args.workdir)), ("output", Path(args.output))):
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".smoke_write_probe"
        probe.write_text("ok")
        probe.unlink()
        notes.append(f"{label} ok")
    return PASS, "; ".join(notes)


def c_disk() -> tuple[str, str]:
    args = parse_args()
    worst = None
    for d in (Path(args.workdir), Path(args.output)):
        d.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(d).free / 1024**3
        worst = free_gb if worst is None else min(worst, free_gb)
    status = PASS if (worst or 0) >= MIN_FREE_GB else WARN
    return status, f"min free {worst:.0f} GB (need ~{MIN_FREE_GB:.0f} GB for 7B)"


def _import(modpath: str) -> str:
    m = importlib.import_module(modpath)
    return getattr(m, "__version__", "ok")


def c_olmo_core() -> tuple[str, str]:
    ver = _import("olmo_core")
    importlib.import_module("olmo_core.nn.transformer")  # pulls TE + flash-attn; exercises nvrtc fix
    return PASS, f"olmo_core {ver} (nn.transformer imports)"


def c_flash_attn() -> tuple[str, str]:
    return PASS, f"flash_attn {_import('flash_attn')}"


def c_transformer_engine() -> tuple[str, str]:
    return PASS, f"transformer_engine {_import('transformer_engine')}"


def c_ring_flash_attn() -> tuple[str, str]:
    importlib.import_module("ring_flash_attn")  # the transformers-5.x shim must hold
    return PASS, "ring_flash_attn imports (transformers shim ok)"


def c_libnvrtc() -> tuple[str, str]:
    rc, out = sh(["bash", "-lc", "ldconfig -p | grep -c 'libnvrtc.so'"])
    n = out.strip() if rc == 0 else "0"
    return (PASS if n not in ("", "0") else WARN), f"libnvrtc on ld path: {n} entrie(s)"


def c_converter_runs() -> tuple[str, str]:
    rc, out = sh([sys.executable, str(CONVERTERS / "convert_checkpoint_to_hf.py"), "--help"], timeout=120)
    return (PASS if rc == 0 else FAIL), "convert_to_hf --help imports cleanly" if rc == 0 else out[:160]


def c_hf_connectivity() -> tuple[str, str]:
    from huggingface_hub import HfApi
    info = HfApi(token=os.environ.get("HF_TOKEN")).model_info("allenai/Olmo-3-7B-Think")
    return PASS, f"reached hub; base model siblings={len(info.siblings or [])}"


def c_torchrun() -> tuple[str, str]:
    rc, _ = sh(["bash", "-lc", "command -v torchrun"])
    return (PASS if rc == 0 else FAIL), "torchrun on PATH" if rc == 0 else "torchrun missing"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fields container smoke test")
    p.add_argument("--workdir", default=os.environ.get("FIELDS_WORKDIR", "/tmp/fields-olmo-sft"))
    p.add_argument("--output", default=os.environ.get("FIELDS_OUTPUT", "./output"))
    p.add_argument("--skip_network", action="store_true", help="Skip the HuggingFace hub reachability check.")
    # parse_known_args so the function is reusable from inside checks without arg conflicts
    return p.parse_known_args(argv)[0]


def main(argv: Optional[list[str]] = None) -> int:
    load_secrets()
    args = parse_args(argv)
    print("=" * 78)
    print("Fields container smoke test")
    print(f"  workdir={args.workdir}  output={args.output}  OLMo-core={OLMO_CORE_ROOT}")
    print("=" * 78)

    # (name, fn, critical)
    checks: list[tuple[str, Callable[[], tuple[str, str]], bool]] = [
        ("python", c_python, False),
        ("nvidia-smi (GPUs visible)", c_nvidia_smi, True),
        ("torch CUDA", c_torch_cuda, True),
        ("FP8 (_scaled_mm)", c_fp8, False),
        ("hf auth whoami", c_hf_whoami, False),
        ("wandb login --verify", c_wandb, False),
        ("core files exist", c_core_files, True),
        ("workdir/output writable", c_writable, True),
        ("free disk", c_disk, False),
        ("libnvrtc linker fix", c_libnvrtc, False),
        ("import olmo_core", c_olmo_core, True),
        ("import flash_attn", c_flash_attn, True),
        ("import transformer_engine", c_transformer_engine, False),
        ("import ring_flash_attn", c_ring_flash_attn, False),
        ("convert_to_hf runs", c_converter_runs, True),
        ("torchrun present", c_torchrun, True),
    ]
    if not args.skip_network:
        checks.append(("HF hub reachable", c_hf_connectivity, False))

    print("\nChecks:")
    for name, fn, critical in checks:
        guard(name, fn, critical=critical)

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_warn = sum(1 for _, s, _ in _results if s == WARN)
    print("\n" + "=" * 78)
    print(f"RESULT: {len(_results)} checks | {n_fail} FAIL | {n_warn} WARN | "
          f"{len(_results) - n_fail - n_warn} PASS")
    print("=" * 78)
    if n_fail:
        print("CRITICAL failures present — the container is NOT ready to train.")
        return 1
    if n_warn:
        print("Ready, with warnings (review the WARN lines above).")
    else:
        print("All green — container ready to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
