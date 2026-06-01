#!/bin/bash
# Baked into the DockerHub image as ENTRYPOINT. Pulls the latest code from GitHub at
# container start, then hands off to the repo's entrypoint. This is the ONLY thing baked
# from us into the image — so iterating on configs/scripts is just `git push` (no rebuild).
# No secrets here: HF_TOKEN / WANDB_API_KEY come from the container env (-e), data from
# /data/training (mounted). Override the source via CODE_REPO / CODE_BRANCH / CODE_DIR.
set -euo pipefail
REPO="${CODE_REPO:-https://github.com/hav4ik/aimo-olmo3-sft}"
BRANCH="${CODE_BRANCH:-olmo3-sft}"
DEST="${CODE_DIR:-/workspace/code}"

echo "[bootstrap] pulling ${REPO}@${BRANCH} -> ${DEST}"
if [ -d "$DEST/.git" ]; then
    git -C "$DEST" fetch --depth 1 origin "$BRANCH" && git -C "$DEST" reset --hard "origin/$BRANCH"
else
    git clone --depth 1 -b "$BRANCH" "$REPO" "$DEST"
fi
exec bash "$DEST/entrypoint.sh" "$@"
