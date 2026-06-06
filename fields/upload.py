#!/usr/bin/env python3
"""Fields Model — convert + upload the trained result.

You pass ONLY the train.py output dir (``--output``); this script does the rest:
  1. finds the highest-step OLMo-core distcp checkpoint under it,
  2. converts it to a HuggingFace safetensors model at ``<output>/model`` — with the legacy
     ``rope_scaling`` + ``rope_theta`` config (transformers 4.x / vLLM / sglang compatible) and
     HF-convention sharding (model-0000i-of-0000N + index for >5 GB), and
  3. uploads ``<output>/model`` to our HuggingFace namespace, auto-named
     ``chankhavu/<base>-<YYYYMMDDHHMMSS>`` — the base comes from the run's experiment; ``--hf_dataset``
     only overrides the base name, and any namespace in it is ignored (so it can't be pushed to the
     wrong place or collide).

``--skip-convert`` uploads an existing ``<output>/model`` as-is (e.g. to re-ship without reconverting).

Examples::

    HF_TOKEN=hf_xxx python /app/upload.py --output /results              # -> chankhavu/<experiment>-<ts>
    HF_TOKEN=hf_xxx python /app/upload.py --output /results --hf_dataset olmo3-7b   # chankhavu/olmo3-7b-<ts>
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from secrets_loader import load_secrets  # baked sibling at /app/secrets_loader.py
except ImportError:
    def load_secrets(path: Optional[str] = None) -> Optional[str]:
        return None

# olmo_core source + the HF converter live in the base image.
OLMO_CORE_ROOT = Path(os.environ.get("OLMO_CORE_ROOT", "/workspace/OLMo-core"))
CONVERT_TO_HF = OLMO_CORE_ROOT / "src" / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"
TOKENIZER = "dolma2"
DEFAULT_OUTPUT = os.environ.get("FIELDS_OUTPUT", "/tmp/olmo-sft/output")
# The HF upload ALWAYS lands here, no matter what the caller passes — so organizers can't push it to the
# wrong namespace or collide a name. The repo id is forced to <HF_NAMESPACE>/<base>-<YYYYMMDDHHMMSS>.
HF_NAMESPACE = os.environ.get("FIELDS_HF_NAMESPACE", "chankhavu")

log = logging.getLogger("fields.upload")


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


UPLOAD_MARKER = "upload_successful.txt"


class AlreadyUploaded(Exception):
    """The highest complete checkpoint already carries upload_successful.txt — nothing new to ship."""


def find_final_checkpoint(output: Path, skip_uploaded: bool = True) -> Path:
    """Return the highest-step COMPLETE distcp checkpoint under the output dir — the final one if training
    finished, or the LATEST SURVIVABLE one if it crashed / was stopped early. A checkpoint counts as
    complete only when DCP's model_and_optim/.metadata (written LAST, after every shard) and config.json
    are both present, so a half-written in-progress save at the moment of a crash is skipped, not picked.
    The converter is handed the stepN ROOT (it reads config.json and appends model_and_optim itself).

    If skip_uploaded (default) and the highest complete checkpoint already has UPLOAD_MARKER, raise
    AlreadyUploaded — we deliberately do NOT fall back to an older one (that would re-ship a stale
    checkpoint). Pass skip_uploaded=False to locate it regardless (e.g. just to write the marker)."""
    steps = sorted(output.rglob("step*"), key=_step_num)

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
    """transformers 5.x serializes RoPE as a single `rope_parameters` field; mirror it back to the
    legacy `rope_scaling` + top-level `rope_theta` (the format the base model + transformers 4.57 /
    vLLM / sglang read) so the exported model loads correctly regardless of the inference stack."""
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
    model.safetensors.index.json) — the HF convention for >5 GB models. Touches ONLY the weights;
    config / generation_config / tokenizer are left exactly as the converter wrote them. Low memory:
    tensor sizes come from the header (no load), and only one shard is materialized at a time."""
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


def convert_checkpoint(checkpoint: Path, hf_out: Path, seq_len: int, shard_bytes: int) -> None:
    """distcp checkpoint -> HF safetensors at hf_out, with the legacy rope config + sharding."""
    hf_out.mkdir(parents=True, exist_ok=True)
    log.info("converting %s -> %s", checkpoint, hf_out)
    proc = subprocess.run([sys.executable, str(CONVERT_TO_HF),
                           "-i", str(checkpoint), "-o", str(hf_out),
                           "-s", str(seq_len), "-t", TOKENIZER,
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
def default_base_name(output: Path) -> str:
    """Base repo name from the run's MANIFEST (the experiment), else a safe default."""
    try:
        for line in (output / "MANIFEST.txt").read_text().splitlines():
            if line.startswith("experiment="):
                return line.split("=", 1)[1].strip() or "olmo3-sft"
    except Exception:  # noqa: BLE001
        pass
    return "olmo3-sft"


def hf_repo_id(base: str) -> str:
    """Force the repo to <HF_NAMESPACE>/<base>-<YYYYMMDDHHMMSS>. Any namespace the caller put in `base`
    is stripped (we keep only the last path segment), so the upload always lands in OUR namespace with a
    unique, non-colliding name."""
    name = (base or "olmo3-sft").rstrip("/").split("/")[-1] or "olmo3-sft"
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{HF_NAMESPACE}/{name}-{ts}"


def upload_hf(repo_id: str, repo_type: str, source_dir: Path, path_in_repo: str, private: bool,
              retries: int = 5, backoff: float = 10.0, max_wait: float = 300.0) -> None:
    """Upload a folder to a HuggingFace repo (multipart/resumable, handles large weights), retrying on
    transient failures (network blips, 429 rate-limit, 5xx). upload_folder is resumable — already-pushed
    LFS files are skipped on retry — so re-calling is cheap. Auth/permission errors (401/403) are NOT
    retried (a token problem won't fix itself). Backoff is exponential: backoff * 2**(n-1), capped at max_wait."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to upload to HuggingFace")
    api = HfApi(token=token)
    dest = f"{'datasets' if repo_type == 'dataset' else 'models'}/{repo_id}"
    attempts = max(1, retries)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            log.info("uploading %s -> hf://%s (%s)%s [attempt %d/%d]", source_dir, dest,
                     "private" if private else "public",
                     f" path={path_in_repo}" if path_in_repo else "", attempt, attempts)
            api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(source_dir),
                              path_in_repo=path_in_repo or None)
            log.info("HuggingFace upload complete -> %s", dest)
            return
        except Exception as exc:  # noqa: BLE001 — retry transient hub/network failures
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise RuntimeError(f"HuggingFace auth/permission error ({status}) on {dest} — "
                                   f"check HF_TOKEN; not retrying") from exc
            last_exc = exc
            if attempt >= attempts:
                break
            wait = min(max_wait, backoff * (2 ** (attempt - 1)))
            log.warning("upload attempt %d/%d failed (%s) — retrying in %.0fs", attempt, attempts, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"HuggingFace upload to {dest} failed after {attempts} attempts: {last_exc}")


def write_success_marker(checkpoint: Path, destination: str) -> None:
    """Drop a small upload_successful.txt into the checkpoint dir after a successful convert+upload, so a
    re-run or external orchestrator can tell this checkpoint was already shipped (and to where)."""
    marker = checkpoint / UPLOAD_MARKER
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        marker.write_text(f"uploaded {ts}\ndestination: {destination}\n")
        log.info("wrote upload marker -> %s", marker)
    except Exception as exc:  # noqa: BLE001 — marker must not fail an otherwise-successful upload
        log.warning("could not write upload marker (%s)", exc)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert + upload the Fields training result",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The only required input: the train.py output dir (Fields' --source_dir / --output_path are aliases).
    p.add_argument("--output", "--source_dir", "--output_path", dest="output", default=DEFAULT_OUTPUT,
                   help="train.py output dir. upload.py finds the distcp under it, converts to "
                        "<output>/model, and uploads that.")
    p.add_argument("--skip-convert", "--skip_convert", dest="skip_convert", action="store_true",
                   help="Skip conversion; upload an existing <output>/model as-is.")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=65536,
                   help="max_position_embeddings written into the HF config.")
    p.add_argument("--shard-size", "--shard_size", dest="shard_size", default="5GB",
                   help="Shard the exported safetensors at this size (e.g. 5GB, 4GiB). '0'/'none' = single file.")
    # HF target
    p.add_argument("--hf_dataset", default=os.environ.get("FIELDS_HF_DATASET", ""),
                   help=f"Base NAME for the HF upload — namespace is ignored and forced to "
                        f"{HF_NAMESPACE}/<name>-<YYYYMMDDHHMMSS>. Empty => name from the run's experiment.")
    p.add_argument("--hf_repo_type", default="model", choices=["model", "dataset"], help="HF repo type.")
    p.add_argument("--hf_path_in_repo", default="", help="Subpath inside the HF repo (default: root).")
    p.add_argument("--hf_private", action="store_true", default=False, help="Create the HF repo private.")
    p.add_argument("--hf_public", dest="hf_private", action="store_false", help="Create the HF repo public (default).")
    p.add_argument("--retries", "--upload-retries", dest="retries", type=int,
                   default=int(os.environ.get("FIELDS_UPLOAD_RETRIES", "5")),
                   help="Retry the HF upload this many times on transient failures (rate-limit / network).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_secrets()  # HF_TOKEN from baked SECRETS.json, else env
    args = parse_args(argv)

    output = Path(args.output).resolve()
    hf_model = output / "model"

    checkpoint: Optional[Path] = None
    if not args.skip_convert:
        try:
            checkpoint = find_final_checkpoint(output)
        except AlreadyUploaded as already:
            log.info("latest checkpoint step%s already uploaded (%s present) — nothing new to ship",
                     already, UPLOAD_MARKER)
            return 0
        log.info("final checkpoint: %s", checkpoint)
        convert_checkpoint(checkpoint, hf_model, args.seq_len, parse_size(args.shard_size))

    if not hf_model.is_dir():
        raise FileNotFoundError(f"no HF model at {hf_model} — run without --skip-convert, or check --output")

    # Target: our HuggingFace namespace, auto-named <HF_NAMESPACE>/<base>-<timestamp>.
    repo = hf_repo_id(args.hf_dataset.strip() or default_base_name(output))
    upload_hf(repo, args.hf_repo_type, hf_model, args.hf_path_in_repo, args.hf_private, retries=args.retries)
    destination = f"hf://{'datasets' if args.hf_repo_type == 'dataset' else 'models'}/{repo}"

    # Mark the converted checkpoint as shipped. In --skip-convert mode we didn't locate it above, so
    # find it now (best-effort — the upload already succeeded, the marker must not change the exit code).
    if checkpoint is None:
        try:
            checkpoint = find_final_checkpoint(output, skip_uploaded=False)
        except Exception:  # noqa: BLE001
            checkpoint = None
    if checkpoint is not None:
        write_success_marker(checkpoint, destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
