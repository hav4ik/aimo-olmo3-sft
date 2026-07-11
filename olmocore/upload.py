#!/usr/bin/env python3
"""Checkpoint watchdog: convert each new OLMo-core distcp checkpoint to HuggingFace + upload it.

Ported from the Fields (NII) FP8 pipeline's upload watcher, adapted for the olmocore recipe's
checkpoint layout and a caller-specified HF repo. Two modes:

  * ``--watch``  : background poll loop (run.sh backgrounds this on node 0). Every ``--interval`` s it
                   spawns a single-shot ``upload.py`` (bounded by ``--timeout`` so a wedged ship can't
                   stall the loop) that converts + ships the LATEST complete checkpoint if it isn't
                   already marked. Intermediate checkpoints land under ``step<N>/`` in the repo.
  * one-shot     : convert + upload the latest complete checkpoint once. ``--final`` ships even if
                   already marked and uploads to the repo ROOT (so the repo root == the final model),
                   never bounded by a timeout (the guaranteed end-of-run deliverable).

A checkpoint is only shipped once — an ``upload_successful.txt`` marker is written into the step dir.
The distcp->HF convert reuses OLMo-core's convert_checkpoint_to_hf.py, which goes through the
sink-aware save_hf_model, so the per-head attention sinks are preserved in the exported model.

    HF_TOKEN=hf_xxx python upload.py --output <ckpt-root> --repo user/olmo3-32b-sft --final
    HF_TOKEN=hf_xxx python upload.py --output <ckpt-root> --repo user/olmo3-32b-sft --watch --interval 300
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Default tokenizer for the to-HF converter. NOTE: convert_checkpoint_to_hf.py feeds -t straight into
# AutoTokenizer.from_pretrained (no alias table like the from-HF path), so the bare alias "dolma2" is NOT
# resolvable — use the real HF repo. Overridden by --tokenizer / OLMO_UPLOAD_TOKENIZER (run.sh passes the
# model's own tokenizer id when OLMO_HF_TOKENIZER=<id>).
DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer"

OLMO_CORE_ROOT = Path(os.environ.get("OLMO_CORE_ROOT", "/workspace/OLMo-core"))
CONVERT_TO_HF = OLMO_CORE_ROOT / "src" / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"
UPLOAD_MARKER = "upload_successful.txt"

log = logging.getLogger("olmocore.upload")


# --------------------------------------------------------------------------------------------------
# Convert: distcp checkpoint -> HuggingFace safetensors
# --------------------------------------------------------------------------------------------------
def parse_size(s: str) -> int:
    """Parse a shard size like '5GB' / '4GiB' / '0' into bytes (0 => disabled). GB=10^9, GiB=2^30."""
    s = (s or "").strip().upper()
    if s in ("", "0", "NONE", "OFF"):
        return 0
    for suf, mult in (("GIB", 2**30), ("MIB", 2**20), ("GB", 10**9), ("MB", 10**6), ("B", 1)):
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * mult)
    return int(s)


def _step_num(p: Path) -> int:
    digits = re.sub(r"\D", "", p.name)
    return int(digits) if digits else -1


class AlreadyUploaded(Exception):
    """The highest complete checkpoint already carries upload_successful.txt — nothing new to ship."""


def find_final_checkpoint(output: Path, skip_uploaded: bool = True, run_name: str = "") -> Path:
    """Return the highest-step COMPLETE distcp checkpoint under output. Complete = model_and_optim/.metadata
    (DCP writes it LAST, after every shard) + config.json both present, so a half-written save at a crash is
    skipped, not picked. The converter is handed the stepN ROOT (it appends model_and_optim itself).

    If skip_uploaded and the highest complete checkpoint already has UPLOAD_MARKER, raise AlreadyUploaded —
    we deliberately do NOT fall back to an older one (that would re-ship a stale checkpoint).

    run_name scopes the search to THIS run's save_folder (…/olmo-sft/<run_name>/step*) — critical on a
    shared volume where a PRIOR run's higher-numbered step* would otherwise be picked and shipped."""
    steps = sorted(output.rglob("step*"), key=_step_num)
    if run_name:
        sep = f"{os.sep}olmo-sft{os.sep}{run_name}{os.sep}"
        scoped = [s for s in steps if sep in f"{s}{os.sep}"]
        if scoped:
            steps = scoped
        else:  # run_name given but no matching path yet (first save not landed) — treat as "nothing yet"
            steps = []

    def has_config(s: Path) -> bool:
        return (s / "config.json").exists()

    complete = [s for s in steps if s.is_dir() and (s / "model_and_optim" / ".metadata").exists() and has_config(s)]
    if not complete:  # fall back to dir-exists for atypical layouts, so we don't wrongly give up
        complete = [s for s in steps if s.is_dir() and (s / "model_and_optim").is_dir() and has_config(s)]
    if not complete:
        if steps:
            raise FileNotFoundError(f"step dirs exist under {output} but none are complete "
                                    f"(need model_and_optim/ + config.json) — did a checkpoint finish saving?")
        raise FileNotFoundError(f"no step* checkpoint under {output} (did training save one?)")

    chosen = complete[-1]
    if skip_uploaded and (chosen / UPLOAD_MARKER).exists():
        raise AlreadyUploaded(chosen.name)
    skipped = [s.name for s in steps if _step_num(s) > _step_num(chosen)]
    if skipped:
        log.warning("using %s (latest COMPLETE checkpoint); skipped later incomplete one(s): %s",
                    chosen.name, ", ".join(skipped))
    return chosen


def legacy_rope_config(config_path: Path) -> None:
    """transformers 5.x serializes RoPE as a single `rope_parameters` field; mirror it back to the legacy
    `rope_scaling` + top-level `rope_theta` (what transformers 4.x / vLLM / sglang read) so the exported
    model loads correctly regardless of the inference stack."""
    try:
        cfg = json.loads(config_path.read_text())
        rp = cfg.pop("rope_parameters", None)
        if rp and not cfg.get("rope_scaling"):
            cfg["rope_theta"] = rp.get("rope_theta", cfg.get("rope_theta", 500000))
            cfg["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
            config_path.write_text(json.dumps(cfg, indent=2))
            log.info("config: mirrored rope_parameters -> rope_scaling + rope_theta (transformers 4.x/vLLM compat)")
    except Exception as exc:  # noqa: BLE001 — config tweak must not fail the export
        log.warning("could not normalize rope config (%s); leaving as-is", exc)


def shard_safetensors(out_dir: Path, max_shard_bytes: int) -> None:
    """Re-split a single model.safetensors into HF-style shards (model-0000i-of-0000N.safetensors +
    model.safetensors.index.json) — the HF convention for >5 GB models (the 32B needs it). Touches ONLY the
    weights; low memory (sizes from the header, one shard materialized at a time)."""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file

        single = out_dir / "model.safetensors"
        if not single.is_file():
            return  # already sharded / nothing to do
        with open(single, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        sizes = {k: v["data_offsets"][1] - v["data_offsets"][0]
                 for k, v in header.items() if k != "__metadata__"}
        shards: list[list[str]] = []
        cur: list[str] = []
        cur_sz = 0
        for name, sz in sizes.items():  # greedy pack in file order
            if cur and cur_sz + sz > max_shard_bytes:
                shards.append(cur)
                cur, cur_sz = [], 0
            cur.append(name)
            cur_sz += sz
        if cur:
            shards.append(cur)
        if len(shards) <= 1:
            log.info("export fits in one shard (<= %d B); leaving single file", max_shard_bytes)
            return
        n = len(shards)
        weight_map: dict[str, str] = {}
        with safe_open(str(single), framework="pt") as f:
            for i, names in enumerate(shards, start=1):
                fname = f"model-{i:05d}-of-{n:05d}.safetensors"
                save_file({name: f.get_tensor(name) for name in names},
                          str(out_dir / fname), metadata={"format": "pt"})
                weight_map.update({name: fname for name in names})
        index = {"metadata": {"total_size": sum(sizes.values())}, "weight_map": weight_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        single.unlink()
        log.info("sharded model.safetensors -> %d shards (<=%d B each)", n, max_shard_bytes)
    except Exception as exc:  # noqa: BLE001 — sharding must not fail the export
        log.warning("could not shard safetensors (%s); leaving single file", exc)


def convert_checkpoint(checkpoint: Path, hf_out: Path, seq_len: int, tokenizer: str, shard_bytes: int) -> None:
    """distcp checkpoint -> HF safetensors at hf_out, sink-preserving, with legacy rope + sharding."""
    # Clean any prior (possibly partial/re-sharded) export so orphan model-*-of-*.safetensors from a
    # failed attempt can't get shipped alongside a fresh index.json.
    shutil.rmtree(hf_out, ignore_errors=True)
    hf_out.mkdir(parents=True, exist_ok=True)
    log.info("converting %s -> %s", checkpoint, hf_out)
    proc = subprocess.run([sys.executable, str(CONVERT_TO_HF),
                           "-i", str(checkpoint), "-o", str(hf_out),
                           "-s", str(seq_len), "-t", tokenizer,
                           "--dtype", "bfloat16", "--skip-validation"])
    if proc.returncode != 0:
        raise RuntimeError(f"convert_checkpoint_to_hf failed (exit {proc.returncode})")
    legacy_rope_config(hf_out / "config.json")
    if shard_bytes:
        shard_safetensors(hf_out, shard_bytes)
    log.info("HF model ready at %s", hf_out)


# --------------------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------------------
def upload_hf(repo_id: str, source_dir: Path, path_in_repo: str, private: bool,
              retries: int = 5, backoff: float = 10.0, max_wait: float = 300.0) -> None:
    """Upload a folder to a HF model repo (multipart/resumable), retrying transient failures (network,
    429, 5xx). upload_folder is resumable — already-pushed LFS files are skipped on retry. Auth errors
    (401/403) are NOT retried. Backoff exponential: backoff * 2**(n-1), capped at max_wait."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to upload to HuggingFace")
    api = HfApi(token=token)
    attempts = max(1, retries)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            log.info("uploading %s -> hf://%s (%s)%s [attempt %d/%d]", source_dir, repo_id,
                     "private" if private else "public",
                     f" path={path_in_repo}" if path_in_repo else "", attempt, attempts)
            api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(source_dir),
                              path_in_repo=path_in_repo or None)
            log.info("HuggingFace upload complete -> hf://%s%s", repo_id,
                     f"/{path_in_repo}" if path_in_repo else "")
            return
        except Exception as exc:  # noqa: BLE001 — retry transient hub/network failures
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise RuntimeError(f"HuggingFace auth/permission error ({status}) on {repo_id} — "
                                   f"check HF_TOKEN has WRITE scope; not retrying") from exc
            last_exc = exc
            if attempt >= attempts:
                break
            wait = min(max_wait, backoff * (2 ** (attempt - 1)))
            log.warning("upload attempt %d/%d failed (%s) — retrying in %.0fs", attempt, attempts, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"HuggingFace upload to {repo_id} failed after {attempts} attempts: {last_exc}")


def write_success_marker(checkpoint: Path, destination: str) -> None:
    """Drop upload_successful.txt into the checkpoint dir so a re-poll knows it was already shipped."""
    marker = checkpoint / UPLOAD_MARKER
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        marker.write_text(f"uploaded {ts}\ndestination: {destination}\n")
        log.info("wrote upload marker -> %s", marker)
    except Exception as exc:  # noqa: BLE001 — marker must not fail an otherwise-successful upload
        log.error("could not write upload marker %s (%s) — this checkpoint may be re-uploaded next poll",
                  marker, exc)


# --------------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------------
def ship_once(args: argparse.Namespace) -> int:
    """Convert + upload the latest complete checkpoint once. --final ships even if marked, to the repo
    ROOT (repo root == final model); otherwise ships intermediates under step<N>/."""
    output = Path(args.output).resolve()
    try:
        checkpoint = find_final_checkpoint(output, skip_uploaded=not args.final, run_name=args.run_name)
    except AlreadyUploaded as already:
        log.info("latest checkpoint step%s already uploaded (%s present) — nothing new to ship",
                 already, UPLOAD_MARKER)
        return 0
    except FileNotFoundError as exc:
        log.info("nothing to upload yet: %s", exc)
        return 3 if args.final else 0

    step = _step_num(checkpoint)
    tail = "" if args.final else f"step{step}"
    # Optional path prefix inside the repo: uploads land at <repo>/<prefix>/step<N> (intermediate) and
    # <repo>/<prefix> (final). No prefix -> <repo>/step<N> and <repo> root (loadable directly).
    path_in_repo = "/".join(p for p in (args.prefix.strip("/"), tail) if p)
    hf_model = output / "_hf_export" / (f"final" if args.final else f"step{step}")
    log.info("%scheckpoint: %s -> hf://%s%s", "FINAL " if args.final else "", checkpoint, args.repo,
             f"/{path_in_repo}" if path_in_repo else "")
    convert_checkpoint(checkpoint, hf_model, args.seq_len, args.tokenizer, parse_size(args.shard_size))
    upload_hf(args.repo, hf_model, path_in_repo, args.private, retries=args.retries)
    write_success_marker(checkpoint, f"hf://{args.repo}" + (f"/{path_in_repo}" if path_in_repo else ""))
    return 0


def watch(args: argparse.Namespace) -> int:
    """Poll loop (node 0). Every --interval s, spawn a single-shot upload bounded by --timeout so a wedged
    ship can't stall the loop (killed + retried next poll). Runs until killed by run.sh at end of training;
    the FINAL uncapped upload is done separately by run.sh."""
    output = Path(args.output).resolve()
    log.info("upload watcher started: poll every %ds, %.0f min/ckpt cap, repo=%s, output=%s",
             args.interval, args.timeout / 60.0, args.repo, output)
    single = [sys.executable, str(Path(__file__).resolve()),
              "--output", str(output), "--repo", args.repo, "--seq-len", str(args.seq_len),
              "--tokenizer", args.tokenizer, "--shard-size", args.shard_size, "--retries", str(args.retries)]
    if args.run_name:
        single += ["--run-name", args.run_name]
    if args.prefix:
        single += ["--prefix", args.prefix]
    if args.private:
        single.append("--private")

    current: dict = {"proc": None}

    def _kill_current() -> None:
        p = current["proc"]
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)  # kill the single-shot AND its convert grandchild
            except (ProcessLookupError, PermissionError):
                pass

    def _on_signal(signum, _frame):
        # run.sh sends SIGTERM at end of training (or on Beaker preempt) — reap the in-flight convert so it
        # can't orphan / collide with the FINAL upload, then exit.
        log.info("upload watcher: signal %d — stopping and reaping in-flight upload", signum)
        _kill_current()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    while True:
        time.sleep(args.interval)  # sleep first — no checkpoint exists at step 0
        # Spawn in its OWN session so a timeout (or SIGTERM) kills the single-shot AND its
        # convert_checkpoint_to_hf grandchild — otherwise a wedged 32B CPU convert orphans and RAM stacks.
        proc = subprocess.Popen(single, start_new_session=True)
        current["proc"] = proc
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log.warning("upload exceeded %ss — killing the process group (retry next poll)", args.timeout)
            _kill_current()
            proc.wait()
        except Exception as exc:  # noqa: BLE001 — watcher must survive any single-poll failure
            log.warning("upload watcher iteration failed (%s); retrying next interval", exc)
        finally:
            current["proc"] = None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert + upload OLMo-core distcp checkpoints to HuggingFace",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", required=True, help="Checkpoint root (rglob'd for step* distcp dirs).")
    p.add_argument("--repo", required=True, help="Target HF model repo id, e.g. user/olmo3-32b-sft-128k.")
    p.add_argument("--run-name", "--run_name", dest="run_name", default=os.environ.get("RUN_NAME", ""),
                   help="Scope discovery to …/olmo-sft/<run_name>/step* (avoids picking a prior run's ckpt on a shared volume).")
    p.add_argument("--prefix", default=os.environ.get("OLMO_HF_UPLOAD_PREFIX", "ycchen-olmo32b-ds-sft"),
                   help="Path prefix inside the HF repo: uploads land at <repo>/<prefix>/step<N> and <repo>/<prefix> (final). Default 'ycchen-olmo32b-ds-sft'; pass '' for the repo root.")
    p.add_argument("--watch", action="store_true", help="Background poll loop (converts+ships each new checkpoint).")
    p.add_argument("--final", action="store_true",
                   help="One-shot end-of-run ship: upload the latest checkpoint to the repo ROOT even if marked.")
    p.add_argument("--interval", type=float, default=float(os.environ.get("OLMO_HF_UPLOAD_INTERVAL", "300")),
                   help="Watch mode: seconds between polls.")
    p.add_argument("--timeout", type=float, default=float(os.environ.get("OLMO_HF_UPLOAD_TIMEOUT", "5400")),
                   help="Watch mode: per-checkpoint convert+ship wall-clock cap (s); wedged ship killed + retried.")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int,
                   default=int(os.environ.get("SEQ_LEN", "65536")), help="max_position_embeddings in the HF config.")
    p.add_argument("--tokenizer", default=os.environ.get("OLMO_UPLOAD_TOKENIZER", DEFAULT_TOKENIZER),
                   help="Tokenizer id passed to the to-HF converter (-t); must be a resolvable HF repo (NOT the 'dolma2' alias).")
    p.add_argument("--shard-size", "--shard_size", dest="shard_size", default="5GB",
                   help="Shard exported safetensors at this size (e.g. 5GB); '0'/'none' = single file.")
    p.add_argument("--private", action="store_true",
                   default=os.environ.get("OLMO_HF_UPLOAD_PRIVATE", "1").lower() not in ("0", "false", "no", ""),
                   help="Create the HF repo private (default). Use --public to override.")
    p.add_argument("--public", dest="private", action="store_false", help="Create the HF repo public.")
    p.add_argument("--retries", type=int, default=int(os.environ.get("OLMO_HF_UPLOAD_RETRIES", "5")),
                   help="Retry the HF upload this many times on transient failures.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    if args.watch:
        return watch(args)
    return ship_once(args)


if __name__ == "__main__":
    sys.exit(main())
