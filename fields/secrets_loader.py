#!/usr/bin/env python3
"""Load baked credentials (HF_TOKEN, WANDB_API_KEY) from SECRETS.json, else fall back to the environment.

When present — e.g. baked into the Singularity image at /app/SECRETS.json — this lets the container run
with no ``-e`` secrets at all. The file is deliberately NOT committed to git and NOT included in the
Docker build used for VastAI testing (there the values come from ``docker run -e`` instead). An env var
that is already set wins, so a baked secret can still be overridden at runtime.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("fields.secrets")

# Keys we propagate from the file into the process environment.
SECRET_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN", "WANDB_ENTITY", "WANDB_PROJECT")
DEFAULT_PATHS = ("/app/SECRETS.json",)


def load_secrets(path: Optional[str] = None) -> Optional[str]:
    """Populate os.environ from the first SECRETS.json found. Returns the path used, or None.

    Search order: explicit `path` -> $FIELDS_SECRETS -> /app/SECRETS.json -> SECRETS.json next to this
    module. Existing env vars are never overwritten (explicit runtime values win)."""
    candidates = [path, os.environ.get("FIELDS_SECRETS"), *DEFAULT_PATHS,
                  str(Path(__file__).resolve().parent / "SECRETS.json")]
    for c in candidates:
        if not c or not Path(c).is_file():
            continue
        try:
            data = json.loads(Path(c).read_text())
        except Exception as exc:  # noqa: BLE001 — a malformed file should not crash the run
            log.warning("could not parse secrets file %s: %s", c, exc)
            return None
        applied: list[str] = []
        for k, v in data.items():
            if v and not os.environ.get(k):   # env wins
                os.environ[k] = str(v)
                applied.append(k)
        if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        log.info("loaded secrets from %s (set: %s)", c, ", ".join(applied) or "none new")
        return c
    log.info("no SECRETS.json found; using environment for credentials")
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    used = load_secrets()
    # Report presence WITHOUT printing secret values.
    for k in SECRET_KEYS:
        print(f"  {k}: {'set' if os.environ.get(k) else 'MISSING'}")
