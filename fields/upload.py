#!/usr/bin/env python3
"""Fields Model — result uploader. Pushes the trained artifact to ONE of two targets:

  * a HuggingFace repo  (``--hf_dataset <repo_id>``) — preferred for model weights: HF chunks large
    files automatically (resumable multipart), so a ~14 GB safetensors goes up cleanly; or
  * a presigned AWS S3 URL (``--s3_url``, the Fields-standard target) — the directory is tar.gz'd into
    a single object and HTTP-PUT to the URL. NB a presigned single PUT caps at 5 GB, so prefer HF for
    full model weights.

Target selection: if ``--hf_dataset`` is set it wins; otherwise the S3 URL is used. Per the Fields spec
the upload URL is baked as the default (``--s3_url``) so it never has to be passed separately — set it
in ``DEFAULT_S3_URL`` below (or the ``FIELDS_S3_URL`` env) per submission.

Examples::

    # to HuggingFace (handles the big safetensors)
    HF_TOKEN=hf_xxx python /app/upload.py --hf_dataset chankhavu/olmo3-7b-sft-result --source_dir /results/model

    # to the presigned S3 URL baked as default
    python /app/upload.py --source_dir /results/model
"""
from __future__ import annotations

import argparse
import logging
import os
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

# Bake the presigned PUT URL here (or via FIELDS_S3_URL) so it need not be passed at runtime.
DEFAULT_S3_URL = os.environ.get("FIELDS_S3_URL", "")
DEFAULT_SOURCE_DIR = os.environ.get("FIELDS_SOURCE_DIR", str(Path(os.environ.get("FIELDS_OUTPUT", "./output")) / "model"))
S3_SINGLE_PUT_LIMIT = 5 * 1024**3  # 5 GiB — AWS hard cap for a single (non-multipart) PUT

log = logging.getLogger("fields.upload")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload the Fields training result to S3 or HuggingFace",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s3_url", default=DEFAULT_S3_URL,
                   help="Presigned S3 PUT URL (baked default). Used when --hf_dataset is not set.")
    p.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR,
                   help="Local dir to upload (the exported HF model by default).")
    p.add_argument("--hf_dataset", default=os.environ.get("FIELDS_HF_DATASET", ""),
                   help="HuggingFace repo id to upload to. If set, HF is used instead of S3.")
    p.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model"],
                   help="HF repo type for --hf_dataset.")
    p.add_argument("--hf_path_in_repo", default="",
                   help="Subpath inside the HF repo (default: repo root).")
    p.add_argument("--hf_private", action="store_true", default=True, help="Create the HF repo private.")
    p.add_argument("--hf_public", dest="hf_private", action="store_false", help="Create the HF repo public.")
    p.add_argument("--archive_name", default="", help="Name for the S3 tar.gz (default: <source_dir>.tar.gz).")
    return p.parse_args(argv)


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
    log.info("HuggingFace upload complete: https://huggingface.co/%s/%s",
             "datasets/" + repo_id if repo_type == "dataset" else repo_id, "")


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


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_secrets()  # HF_TOKEN from baked SECRETS.json, else env
    args = parse_args(argv)

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"--source_dir does not exist: {source_dir}")

    if args.hf_dataset:
        upload_hf(args.hf_dataset, args.hf_repo_type, source_dir, args.hf_path_in_repo, args.hf_private)
    elif args.s3_url:
        upload_s3(args.s3_url, source_dir, args.archive_name)
    else:
        raise SystemExit("no target: set --hf_dataset <repo>, or --s3_url (or bake DEFAULT_S3_URL).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
