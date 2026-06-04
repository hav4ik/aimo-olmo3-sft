#!/usr/bin/env python3
"""Fields Model — convert + upload the trained result.

You pass ONLY the train.py output dir (``--output``); this script does the rest:
  1. finds the highest-step OLMo-core distcp checkpoint under it,
  2. converts it to a HuggingFace safetensors model at ``<output>/model`` — with the legacy
     ``rope_scaling`` + ``rope_theta`` config (transformers 4.x / vLLM / sglang compatible) and
     HF-convention sharding (model-0000i-of-0000N + index for >5 GB), and
  3. uploads ``<output>/model``:
       * by DEFAULT to our HuggingFace namespace, auto-named ``chankhavu/<base>-<YYYYMMDDHHMMSS>`` — the
         base comes from the run's experiment; ``--hf_dataset`` only overrides the base name, and any
         namespace in it is ignored (so it can't be pushed to the wrong place or collide); or
       * to a presigned AWS S3 URL via ``--hf_dataset none --s3_url <url>`` (tar.gz + PUT, single ≤ 5 GB).

``--skip-convert`` uploads an existing ``<output>/model`` as-is (e.g. to re-ship without reconverting).

Examples::

    HF_TOKEN=hf_xxx python /app/upload.py --output /results              # -> chankhavu/<experiment>-<ts>
    HF_TOKEN=hf_xxx python /app/upload.py --output /results --hf_dataset olmo3-7b   # chankhavu/olmo3-7b-<ts>
    python /app/upload.py --output /results --hf_dataset none --s3_url https://...  # -> S3
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
import tarfile
import tempfile
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
DEFAULT_OUTPUT = os.environ.get("FIELDS_OUTPUT", "/data/training/output")
DEFAULT_S3_URL = os.environ.get("FIELDS_S3_URL", "")  # bake the presigned PUT URL here (or FIELDS_S3_URL)
# The HF upload ALWAYS lands here, no matter what the caller passes — so organizers can't push it to the
# wrong namespace or collide a name. The repo id is forced to <HF_NAMESPACE>/<base>-<YYYYMMDDHHMMSS>.
HF_NAMESPACE = os.environ.get("FIELDS_HF_NAMESPACE", "chankhavu")
S3_SINGLE_PUT_LIMIT = 5 * 1024**3  # AWS hard cap for a single (non-multipart) PUT

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


def find_final_checkpoint(output: Path) -> Path:
    """Find the highest-step distcp checkpoint ROOT anywhere under the output dir. The converter reads
    <dir>/config.json (per-checkpoint, ConfigSaverCallback) and appends model_and_optim itself, so we
    return the stepN dir that has BOTH."""
    steps = sorted(output.rglob("step*"), key=_step_num)
    complete = [s for s in steps if s.is_dir() and (s / "model_and_optim").is_dir() and (s / "config.json").exists()]
    if complete:
        return complete[-1]
    if steps:
        raise FileNotFoundError(f"checkpoint {steps[-1]} is missing config.json or model_and_optim/")
    raise FileNotFoundError(f"no step* checkpoint under {output} (did training save one?)")


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


def upload_hf(repo_id: str, repo_type: str, source_dir: Path, path_in_repo: str, private: bool) -> None:
    """Upload a folder to a HuggingFace repo (multipart/resumable, handles large weights)."""
    from huggingface_hub import HfApi  # imported here so S3-only runs don't require it

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to upload to HuggingFace")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    log.info("uploading %s -> hf://%s/%s (%s)%s", source_dir, repo_type, repo_id,
             "private" if private else "public", f" path={path_in_repo}" if path_in_repo else "")
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(source_dir),
                      path_in_repo=path_in_repo or None)
    log.info("HuggingFace upload complete -> %s/%s",
             "datasets" if repo_type == "dataset" else "models", repo_id)


def make_tarball(source_dir: Path, archive_path: Path) -> Path:
    """Pack source_dir into a .tar.gz (the single object PUT to the presigned URL)."""
    log.info("packing %s -> %s", source_dir, archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(str(source_dir), arcname=source_dir.name)
    return archive_path


def upload_s3(url: str, source_dir: Path, archive_name: str) -> None:
    """tar.gz the dir and HTTP-PUT it to the presigned S3 URL."""
    import requests  # imported here so HF-only runs don't require it

    with tempfile.TemporaryDirectory() as tmp:
        name = archive_name or f"{source_dir.name}.tar.gz"
        archive = make_tarball(source_dir, Path(tmp) / name)
        size = archive.stat().st_size
        log.info("archive size: %.2f GiB", size / 1024**3)
        if size > S3_SINGLE_PUT_LIMIT:
            log.warning("archive %.2f GiB exceeds the 5 GiB single-PUT presigned limit — this PUT will "
                        "likely fail; use --hf_dataset for large weights, or a multipart presigned URL.",
                        size / 1024**3)
        with archive.open("rb") as fh:
            resp = requests.put(url, data=fh, headers={"Content-Type": "application/gzip"})
        if not resp.ok:
            raise RuntimeError(f"S3 PUT failed: HTTP {resp.status_code} {resp.text[:300]}")
        log.info("S3 upload complete (HTTP %d)", resp.status_code)


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
                        f"{HF_NAMESPACE}/<name>-<YYYYMMDDHHMMSS>. Empty => name from the run's experiment. "
                        "'none' => upload to S3 instead.")
    p.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model"], help="HF repo type.")
    p.add_argument("--hf_path_in_repo", default="", help="Subpath inside the HF repo (default: root).")
    p.add_argument("--hf_private", action="store_true", default=False, help="Create the HF repo private.")
    p.add_argument("--hf_public", dest="hf_private", action="store_false", help="Create the HF repo public (default).")
    # S3 target
    p.add_argument("--s3_url", default=DEFAULT_S3_URL, help="Presigned S3 PUT URL (baked default). Used when --hf_dataset is unset.")
    p.add_argument("--archive_name", default="", help="Name for the S3 tar.gz (default: model.tar.gz).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_secrets()  # HF_TOKEN from baked SECRETS.json, else env
    args = parse_args(argv)

    output = Path(args.output).resolve()
    hf_model = output / "model"

    if not args.skip_convert:
        checkpoint = find_final_checkpoint(output)
        log.info("final checkpoint: %s", checkpoint)
        convert_checkpoint(checkpoint, hf_model, args.seq_len, parse_size(args.shard_size))

    if not hf_model.is_dir():
        raise FileNotFoundError(f"no HF model at {hf_model} — run without --skip-convert, or check --output")

    # Target: our HuggingFace namespace by default, auto-named <HF_NAMESPACE>/<base>-<timestamp>.
    # Pass --hf_dataset none (+ --s3_url) to use S3 instead.
    if args.hf_dataset.strip().lower() in ("none", "off"):
        if not args.s3_url:
            raise SystemExit("HF disabled (--hf_dataset none) but no --s3_url given.")
        upload_s3(args.s3_url, hf_model, args.archive_name)
    else:
        repo = hf_repo_id(args.hf_dataset.strip() or default_base_name(output))
        upload_hf(repo, args.hf_repo_type, hf_model, args.hf_path_in_repo, args.hf_private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
