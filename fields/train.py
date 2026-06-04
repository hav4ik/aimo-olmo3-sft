#!/usr/bin/env python3
"""Fields Model — TRAIN-variant entrypoint for Olmo-3 SFT on OLMo-core.

This is the single ``/app/train.py`` the Singularity image exposes. It is a thin, fully self-
describing orchestrator over our tested ``olmocore/run.sh`` pipeline:

    download (HF base model + tokenized dataset + code)  ->  --workdir
    convert  (HF base -> OLMo-core distcp)               ->  --workdir
    train    (torchrun the Olmo-3 SFT recipe)            ->  --output/<run>/stepN  (distcp)
    export   (final distcp -> HuggingFace safetensors)   ->  --output/model        (the deliverable)

Nothing is baked into the image except code + dependencies: the heavy artifacts (base weights ~15 GB,
tokenized data ~15 GB) are pulled at runtime into ``--workdir`` (a scratch / bind-mounted dir), and all
results (logs + checkpoints + the exported HF model) land under ``--output``.

The whole run is selected by ONE flag: ``--experiment olmo_7b_bf16`` (or ``olmo_7b_fp8``). Every other
knob is an optional CLI override with a default that reproduces AI2's published Olmo-3 SFT recipe.

Examples::

    # zero-config: download everything, train olmo3-7b bf16, export HF model to ./output/model
    python /app/train.py --experiment olmo_7b_bf16 --workdir /scratch --output /results

    # FP8 (rowwise on H200), 8 GPUs, custom LR
    python /app/train.py --experiment olmo_7b_fp8 --num_gpus 8 --learning_rate 5e-5 \
        --workdir /scratch --output /results

    # use host-provided paths instead of downloading
    python /app/train.py --model_path /data/Olmo-3-7B-Think --dataset_path /data/tok --output /results
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from secrets_loader import load_secrets  # baked sibling at /app/secrets_loader.py
except ImportError:  # keep train.py runnable even if the loader is absent
    def load_secrets(path: Optional[str] = None) -> Optional[str]:
        return None

# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------
# Where the heavy deps + olmo_core source live in the image (built by docker/base/Dockerfile.olmo-core-sft).
OLMO_CORE_ROOT = Path(os.environ.get("OLMO_CORE_ROOT", "/workspace/OLMo-core"))
CONVERT_TO_HF = OLMO_CORE_ROOT / "src" / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"

# Baked copy of THIS repo's code (run.sh + sft_scripts), used when --code_repo is empty / clone fails.
BAKED_CODE_ROOT = Path(os.environ.get("FIELDS_CODE_ROOT", "/app/code"))

# The deterministic identity train.py pins so it can locate the trained checkpoint afterwards.
RUN_USER = "fields"  # -> save_folder = {OLMO_SFT_SAVE_ROOT}/checkpoints/{RUN_USER}/olmo-sft/{run_name}
TOKENIZER = "dolma2"
DEFAULT_DATASET_REPO = "chankhavu/smolmo-proofs-cot-sft"
DEFAULT_DATASET_SUBDIR = "olmocore"
DEFAULT_CODE_REPO = "https://github.com/hav4ik/aimo-olmo3-sft"
DEFAULT_CODE_REF = "olmo3-sft"


@dataclasses.dataclass(frozen=True)
class Recipe:
    """A named (size, precision) recipe and the AI2 defaults that reproduce it."""

    model_size: str          # "7b" | "32b"
    precision: str           # "bf16" | "fp8"
    model_repo: str          # default HF base model (overridable by --model_path)
    default_lr: float        # AI2 SFT learning rate for this size
    default_epochs: float = 2.0
    seq_len: int = 65536


RECIPES: dict[str, Recipe] = {
    "olmo_7b_bf16": Recipe("7b", "bf16", "allenai/Olmo-3-7B-Think", 5e-5),
    "olmo_7b_fp8": Recipe("7b", "fp8", "allenai/Olmo-3-7B-Think", 5e-5),
    # 32B recipes are added once the 7B submission passes (lr 1e-4, GBS 4,194,304).
}

# Mirrors olmo-core's MAX_RANK_MICROBATCH_SIZE_TOKENS (the SFT script). cp_degree is auto-derived from
# seq_len against this cap: H100/H200 = 16384 (cluster local_h100). B200 doubles it — not our target.
MAX_TOKENS_PER_RANK = 16384

log = logging.getLogger("fields.train")


def cp_degree(seq_len: int, max_tokens_per_rank: int = MAX_TOKENS_PER_RANK) -> int:
    """Replicate olmo-core BatchSizeConfig: smallest power-of-2 cp with seq_len/cp <= cap (1 if it fits).
    On H200 at seq 65536 this is 4 — see the SFT script's BatchSizeConfig.__post_init__."""
    if seq_len <= max_tokens_per_rank:
        return 1
    cp = 2
    while seq_len // cp > max_tokens_per_rank:
        cp *= 2
    return cp


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the Fields CLI. Every option has a default so the harness can call train.py bare."""
    p = argparse.ArgumentParser(
        description="Olmo-3 SFT (OLMo-core) — Fields train variant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # The one knob that picks the whole recipe.
    p.add_argument("--experiment", default="olmo_7b_bf16", choices=sorted(RECIPES),
                   help="Recipe to run: olmo_<size>_<precision>.")

    # The two roots the user asked for.
    p.add_argument("--workdir", default=os.environ.get("FIELDS_WORKDIR", "/tmp/fields-olmo-sft"),
                   help="Scratch root: base model, tokenized dataset and code are downloaded here.")
    p.add_argument("--output", default=os.environ.get("FIELDS_OUTPUT", "./output"),
                   help="Deliverable root: logs, training checkpoints and the exported HF model.")

    # Fields-standard paths (optional — override the corresponding download when a local path is given).
    p.add_argument("--model_path", default="",
                   help="Local dir of base HF weights. If unset, the experiment's model_repo is downloaded.")
    p.add_argument("--dataset_path", default="",
                   help="Local dir with token_ids_part_*.npy. If unset, the dataset repo is downloaded.")
    p.add_argument("--output_path", default="",
                   help="Alias for --output (Fields-standard name). Takes precedence over --output if set.")
    p.add_argument("--logdir", default="",
                   help="Log dir. Defaults to <output>/logs.")

    # Optional hyperparameters (0/empty => recipe default => reproduces AI2). Fields-standard names.
    p.add_argument("--num_gpus", "--num-gpus", dest="num_gpus", type=int, default=0,
                   help="GPUs per node (0 => all visible).")
    p.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=0.0,
                   help="Peak LR (0 => recipe default).")
    p.add_argument("--num_train_epochs", "--num-train-epochs", dest="num_train_epochs", type=float, default=0.0,
                   help="Epochs (0 => recipe default = 2).")

    # Batching — the clean inputs olmo-core derives cp_degree / rank-microbatch / grad-accum FROM.
    p.add_argument("--global-batch-tokens", "--global_batch_tokens", dest="global_batch_tokens",
                   type=int, default=1_048_576,
                   help="Global batch in TOKENS. olmo-core derives cp_degree/rank-microbatch/grad-accum from "
                        "this + seq_len + world_size. Default 1,048,576 (AI2 7B recipe).")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=0,
                   help="Max sequence length (0 => recipe default = 65536).")
    p.add_argument("--rank-microbatch-tokens", "--rank_microbatch_tokens", dest="rank_microbatch_tokens",
                   type=int, default=0,
                   help="ADVANCED: per-DP-rank microbatch in TOKENS — MUST be a multiple of seq_len (per-GPU "
                        "tokens = this / cp_degree). 0 => olmo-core auto-derives (one sequence/rank). Raise to "
                        "pack more sequences per microstep and spend spare VRAM on throughput.")
    p.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=0,
                   help="Cap training at N steps (0 => use epochs).")

    # Recipe internals — clean env pass-through to run.sh / the SFT script (an existing env value wins).
    p.add_argument("--run-suffix", "--run_suffix", dest="run_suffix", default=os.environ.get("RUN_SUFFIX", ""),
                   help="Suffix appended to RUN_NAME (W&B run + checkpoint dir). Default: none.")
    p.add_argument("--olmo-ac-budget", "--olmo_ac_budget", dest="olmo_ac_budget",
                   default=os.environ.get("OLMO_AC_BUDGET", "0.8"),
                   help="Activation-checkpointing budget: 1.0=store all (max mem), 0.0=recompute all. Default 0.8.")
    p.add_argument("--olmo-fused-lce", "--olmo_fused_lce", dest="olmo_fused_lce",
                   default=os.environ.get("OLMO_FUSED_LCE", "1"),
                   help="Liger fused linear cross-entropy (z-loss fix is in the fork). Default 1 (on).")
    p.add_argument("--olmo-optim-dtype", "--olmo_optim_dtype", dest="olmo_optim_dtype",
                   default=os.environ.get("OLMO_OPTIM_DTYPE", "bf16"),
                   help="Optimizer-state dtype (bf16 halves optim memory; applies to skip_step). Default bf16.")

    # Checkpoint cadence + retention (distcp ~100 GB/7B each — keep these sane or disk blows up).
    p.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=1000,
                   help="PERSISTENT checkpoint every N steps (kept = --keep-last, so disk stays bounded). Default 1000.")
    p.add_argument("--ephemeral-interval", "--ephemeral_interval", dest="ephemeral_interval", type=int,
                   default=500, help="Ephemeral (rotating, only-latest-kept) resume checkpoint every N steps. Default 500.")
    p.add_argument("--keep-last", "--keep_last", dest="keep_last", type=int, default=3,
                   help="Cap on PERSISTENT checkpoints kept (oldest pruned as new ones land; 0 = keep all). Default 3.")

    # Sources (override the defaults if you host the artifacts elsewhere).
    p.add_argument("--dataset_repo", default=DEFAULT_DATASET_REPO, help="HF dataset repo to download when --dataset_path is unset.")
    p.add_argument("--dataset_subdir", default=DEFAULT_DATASET_SUBDIR, help="Subdir in the dataset repo holding the .npy.")
    p.add_argument("--code_repo", default=DEFAULT_CODE_REPO, help="Code repo to clone (empty => use baked /app/code).")
    p.add_argument("--code_ref", default=DEFAULT_CODE_REF, help="Code branch/tag/commit.")
    p.add_argument("--skip_export", action="store_true", help="Train only; do not export the HF model.")
    return p.parse_args(argv)


def setup_logging(logdir: Path) -> None:
    """Tee logs to stdout and to <logdir>/train.py.log (Fields requires all logs under logdir)."""
    logdir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout),
                                       logging.FileHandler(logdir / "train.py.log")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=handlers, force=True)


def run(cmd: list[str], *, env: Optional[dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
    """Run a subprocess, streaming its output; raise with context on failure."""
    log.info("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def stage_code(code_repo: str, code_ref: str, workdir: Path) -> Path:
    """Clone the code repo into <workdir>/code (so code iterates without rebuilding the image);
    fall back to the baked copy if cloning is disabled or fails. Returns the code root."""
    if not code_repo:
        log.info("using baked code at %s", BAKED_CODE_ROOT)
        return BAKED_CODE_ROOT
    dest = workdir / "code"
    try:
        if (dest / ".git").is_dir():
            run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", code_ref])
            run(["git", "-C", str(dest), "checkout", "-f", "FETCH_HEAD"])
        else:
            run(["git", "clone", "--depth", "1", "--branch", code_ref, code_repo, str(dest)])
        sha = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "--short", "HEAD"]).decode().strip()
        log.info("code: %s @ %s (%s)", code_ref, sha, code_repo)
        return dest
    except Exception as exc:  # noqa: BLE001 — any clone failure should degrade to baked code
        if BAKED_CODE_ROOT.exists():
            log.warning("code clone failed (%s); falling back to baked %s", exc, BAKED_CODE_ROOT)
            return BAKED_CODE_ROOT
        raise


def build_env(args: argparse.Namespace, recipe: Recipe, workdir: Path, output: Path,
              logdir: Path, seq_len: int, ckpt_base: Path, save_root: Path) -> dict[str, str]:
    """Translate the Fields CLI + recipe into the env contract olmocore/run.sh understands."""
    env = dict(os.environ)
    env.update(
        # roots
        DATA=str(workdir),                      # downloads + datasets + scratch
        OLMO_SFT_SAVE_ROOT=str(save_root),      # training checkpoints + work_dir  (under --output)
        CKPT=str(ckpt_base / "model_and_optim"),  # base HF->distcp conversion (in --workdir)
        USER=RUN_USER,                          # pins save_folder/{user}/ so we can find the checkpoint
        # recipe selection
        MODEL_SIZE=recipe.model_size,
        PRECISION=recipe.precision,
        RUN_NAME=run_name(recipe),
        SEQ_LEN=str(seq_len),
        EPOCHS=str(args.num_train_epochs or recipe.default_epochs),
        LR=str(args.learning_rate or recipe.default_lr),
        # topology: one node, this many GPUs (multi-node still works via MASTER_ADDR/WORLD_SIZE env)
        NPROC_PER_NODE=str(args.num_gpus or visible_gpus()),
    )
    # data source: local path wins, else let run.sh download from HF into DATA
    if args.dataset_path:
        env["DATASET"] = args.dataset_path
    else:
        env["DATASET_HF"] = args.dataset_repo
        env["DATASET_SUBDIR"] = args.dataset_subdir
        env["DATASET_NAME"] = "fields"
    # base model source: local dir wins, else the recipe's HF repo (run.sh stages it)
    env["HF_MODEL"] = args.model_path or recipe.model_repo
    # batching: GLOBAL_BATCH_SIZE is the primary knob (olmo-core derives cp/microbatch/grad-accum from
    # it). RANK_MICROBATCH_TOKENS is the optional VRAM/throughput override (0 => olmo-core auto-derives
    # the minimal 1 sequence/rank). Both in tokens.
    env["GLOBAL_BATCH_SIZE"] = str(args.global_batch_tokens)
    if args.rank_microbatch_tokens:
        env["RANK_MICROBATCH_TOKENS"] = str(args.rank_microbatch_tokens)
    if args.max_steps:
        env["MAX_STEPS"] = str(args.max_steps)
    # recipe internals (pass-through to run.sh / the SFT script)
    env["OLMO_AC_BUDGET"] = str(args.olmo_ac_budget)
    env["OLMO_FUSED_LCE"] = str(args.olmo_fused_lce)
    env["OLMO_OPTIM_DTYPE"] = str(args.olmo_optim_dtype)
    env["OLMO_SAVE_INTERVAL"] = str(args.save_interval)
    env["OLMO_EPHEMERAL_INTERVAL"] = str(args.ephemeral_interval)
    env["OLMO_KEEP_LAST_CKPTS"] = str(args.keep_last)
    if args.run_suffix:
        env["RUN_SUFFIX"] = str(args.run_suffix)
    # logs: run.sh + torchrun inherit; we also tee bootstrap-style below
    env["LOGDIR"] = str(logdir)
    return env


def run_name(recipe: Recipe) -> str:
    return f"olmo3-{recipe.model_size}-sft-{recipe.precision}"


def visible_gpus() -> int:
    """GPU count without importing torch (keeps train.py importable on a login node)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode()
        return max(1, len([ln for ln in out.splitlines() if ln.strip()]))
    except Exception:  # noqa: BLE001
        return 1


def find_final_checkpoint(save_root: Path, recipe: Recipe) -> Path:
    """Locate the highest-step trained distcp checkpoint to feed convert_to_hf -i.

    Layout (from the SFT script): {save_root}/checkpoints/{user}/olmo-sft/{run_name}/stepN[/model_and_optim].
    Globs defensively (the {user}/ segment and the model_and_optim/ subdir can vary)."""
    base = save_root / "checkpoints" / RUN_USER / "olmo-sft" / run_name(recipe)
    candidates = sorted(base.glob("step*"), key=lambda p: int(re.sub(r"\D", "", p.name) or -1))
    if not candidates:  # fall back to a broad search if the {user} segment differs
        candidates = sorted(save_root.rglob("step*"), key=lambda p: int(re.sub(r"\D", "", p.name) or -1))
    if not candidates:
        raise FileNotFoundError(f"no step* checkpoint under {save_root} (training may have failed to save)")
    step = candidates[-1]
    for sub in ("model_and_optim", "model"):
        if (step / sub).is_dir():
            return step / sub
    return step


def export_hf(checkpoint: Path, out_dir: Path, seq_len: int) -> None:
    """Convert the trained OLMo-core distcp checkpoint to a HuggingFace safetensors model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(CONVERT_TO_HF),
         "-i", str(checkpoint), "-o", str(out_dir),
         "-s", str(seq_len), "-t", TOKENIZER,
         "--dtype", "bfloat16", "--skip-validation"])
    log.info("exported HF model -> %s", out_dir)


# --------------------------------------------------------------------------------------------------
# W&B continuity — open the run at setup, hand it off to olmo-core's rank-0 worker (same run id)
# --------------------------------------------------------------------------------------------------
# run.sh markers (substring -> setup phase) used to update the run summary while download/convert run.
SETUP_MARKERS = [
    ("[olmocore] staging HF model", "download_model"),
    ("[olmocore] converting", "convert"),
    ("[olmocore] convert complete", "convert_done"),
    ("[olmocore] downloading", "download_data"),
    ("[olmocore] data ready", "data_done"),
]
TRAIN_LAUNCH_MARKER = "[olmocore] run:"  # printed right before torchrun => hand the run to the worker


def is_node_zero() -> bool:
    """Only node 0 logs to W&B (matches olmo-core, which logs from global rank 0 = node-0's worker)."""
    return os.environ.get("NODE_RANK", os.environ.get("GLOBAL_RANK", "0")) == "0"


def wandb_run_id(name: str) -> str:
    """Deterministic W&B run id from the run name (so a crash-relaunch resumes the same run)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:128]


def open_setup_wandb(args: argparse.Namespace, recipe: Recipe, final_name: str, seq_len: int, cp: int,
                     rank_microbatch: int, workdir: Path, output: Path, code_sha: str):
    """Open the W&B run NOW (node 0 only) so the job is visible during download/convert, and export the
    shared id/resume env so olmo-core's rank-0 worker resumes THIS run for training. Returns the run or None."""
    if not is_node_zero() or not os.environ.get("WANDB_API_KEY"):
        return None
    run_id = wandb_run_id(final_name)
    project = os.environ.setdefault("WANDB_PROJECT", "olmo3-7b-sft")  # match olmo-core's default
    os.environ["WANDB_RUN_ID"] = run_id      # inherited by run.sh -> torchrun -> the rank-0 worker
    os.environ["WANDB_RESUME"] = "allow"
    try:
        import wandb
        wdir = output / "wandb"
        wdir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            id=run_id, resume="allow", project=project, name=final_name,
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=os.environ.get("WANDB_RUN_GROUP") or None, dir=str(wdir),
            config={
                "experiment": args.experiment, "model_size": recipe.model_size,
                "precision": recipe.precision, "seq_len": seq_len, "cp_degree": cp,
                "global_batch_tokens": args.global_batch_tokens,
                "rank_microbatch_tokens": args.rank_microbatch_tokens or rank_microbatch,
                "num_gpus": args.num_gpus or visible_gpus(), "ac_budget": args.olmo_ac_budget,
                "fused_lce": args.olmo_fused_lce, "optim_dtype": args.olmo_optim_dtype,
                "code_sha": code_sha, "workdir": str(workdir), "output": str(output),
            },
        )
        run.summary["setup/phase"] = "starting"
        log.info("W&B run open (id=%s, project=%s) — the training worker resumes this same run", run_id, project)
        return run
    except Exception as exc:  # noqa: BLE001 — never let W&B block training
        log.warning("could not open early W&B run (%s); continuing without it", exc)
        return None


def stream_run_sh(cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path, wb) -> int:
    """Run run.sh, tee its output to console + <logdir>/run.log, and (if wb) update the run summary on
    setup markers — finishing the run at the training-launch marker so the worker can resume it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    active = wb is not None
    log.info("exec: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log_path, "a") as lf:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            if active:
                try:
                    for sub, phase in SETUP_MARKERS:
                        if sub in line:
                            wb.summary[f"setup/{phase}_at_sec"] = round(time.monotonic() - t0, 1)
                            wb.summary["setup/phase"] = phase
                    if TRAIN_LAUNCH_MARKER in line:
                        wb.summary["setup/total_sec"] = round(time.monotonic() - t0, 1)
                        wb.summary["setup/phase"] = "training"
                        wb.finish()        # hand off: the rank-0 worker resumes this run id
                        active = False
                except Exception as exc:  # noqa: BLE001
                    log.warning("W&B setup logging error (%s); dropping early logging", exc)
                    active = False
    proc.wait()
    if active:  # run.sh exited before training started (e.g. a setup crash) — close the run cleanly
        try:
            wb.summary["setup/phase"] = "failed_before_training"
            wb.finish()
        except Exception:  # noqa: BLE001
            pass
    return proc.returncode


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    load_secrets()  # HF_TOKEN / WANDB_API_KEY from baked SECRETS.json, else env
    recipe = RECIPES[args.experiment]
    seq_len = args.seq_len or recipe.seq_len

    output = Path(args.output_path or args.output).resolve()
    workdir = Path(args.workdir).resolve()
    logdir = Path(args.logdir).resolve() if args.logdir else output / "logs"
    save_root = output / "internal"          # training checkpoints + work_dir live here
    ckpt_base = workdir / "base-distcp"       # one-time HF->distcp conversion of the base model
    hf_out = output / "model"                 # THE deliverable: exported HF safetensors

    workdir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(logdir)

    t0 = time.monotonic()
    log.info("Fields Olmo-3 SFT | experiment=%s size=%s precision=%s seq_len=%d",
             args.experiment, recipe.model_size, recipe.precision, seq_len)
    log.info("workdir=%s  output=%s  logdir=%s", workdir, output, logdir)

    # Batching summary (informational; olmo-core does the real derivation). rank-microbatch must be a
    # multiple of seq_len if set — fail early with a clear message rather than mid-training.
    cp = cp_degree(seq_len)
    rank_microbatch = args.rank_microbatch_tokens or seq_len  # auto = 1 sequence/rank
    if args.rank_microbatch_tokens and args.rank_microbatch_tokens % seq_len != 0:
        raise ValueError(f"--rank-microbatch-tokens ({args.rank_microbatch_tokens}) must be a multiple of "
                         f"seq_len ({seq_len}) — olmo-core requires it.")
    rmb = (f"{args.rank_microbatch_tokens} tok ({args.rank_microbatch_tokens // seq_len} seq/rank, "
           f"~{args.rank_microbatch_tokens // cp} tok/GPU)" if args.rank_microbatch_tokens
           else f"auto (1 seq/rank, ~{seq_len // cp} tok/GPU)")
    log.info("batching | global=%d tok | seq_len=%d | cp_degree~%d | rank_microbatch=%s | ac_budget=%s",
             args.global_batch_tokens, seq_len, cp, rmb, args.olmo_ac_budget)

    code_root = stage_code(args.code_repo, args.code_ref, workdir)
    run_sh = code_root / "olmocore" / "run.sh"
    if not run_sh.is_file():
        raise FileNotFoundError(f"run.sh not found at {run_sh}")
    try:
        code_sha = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        code_sha = "baked"

    # Open the W&B run now (node 0) so the job is visible from setup; export the shared id/resume so
    # olmo-core's rank-0 worker RESUMES this same run for training metrics (one continuous timeline).
    final_name = run_name(recipe) + (f"-{args.run_suffix}" if args.run_suffix else "")
    wb = open_setup_wandb(args, recipe, final_name, seq_len, cp, rank_microbatch, workdir, output, code_sha)

    env = build_env(args, recipe, workdir, output, logdir, seq_len, ckpt_base, save_root)

    # run.sh does: HF base -> distcp convert (cached) + dataset download + torchrun training. Streamed so
    # we can update the W&B setup phases from run.sh's markers and hand the run off at training launch.
    rc = stream_run_sh(["bash", str(run_sh)], env, code_root, logdir / "run.log", wb)
    if rc != 0:
        raise RuntimeError(f"run.sh failed (exit {rc}) — see {logdir / 'run.log'}")
    log.info("training finished in %.1f min", (time.monotonic() - t0) / 60.0)

    if args.skip_export:
        log.info("--skip_export set; leaving distcp checkpoints under %s", save_root)
        return 0

    checkpoint = find_final_checkpoint(save_root, recipe)
    log.info("final checkpoint: %s", checkpoint)
    export_hf(checkpoint, hf_out, seq_len)

    # Manifest so the deliverable is self-describing.
    (output / "MANIFEST.txt").write_text(
        f"experiment={args.experiment}\nmodel_size={recipe.model_size}\nprecision={recipe.precision}\n"
        f"seq_len={seq_len}\nsource_checkpoint={checkpoint}\nhf_model={hf_out}\n"
        f"elapsed_min={(time.monotonic() - t0) / 60.0:.1f}\n")
    log.info("DONE in %.1f min | HF model: %s", (time.monotonic() - t0) / 60.0, hf_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
