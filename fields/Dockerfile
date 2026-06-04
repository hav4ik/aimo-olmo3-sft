# Fields Model deploy image — bakes the /app entrypoints; everything heavy (base model + tokenized
# dataset) is downloaded at RUNTIME into --workdir. Built as a thin layer on the olmocore deploy image
# (olmo_core source + FA2/FA3 + TransformerEngine + the libnvrtc fix + the ring-flash-attn shim).
#
# Build context = repo root:
#   docker build -f fields/Dockerfile -t chankhavu/olmo3-fields:cu130 .
#
# Vast.AI (Docker): pass credentials with `-e HF_TOKEN=... -e WANDB_API_KEY=...`. Do NOT bake
# SECRETS.json into a PUSHED image — it would leak. The SECRETS.json bake is for the Singularity .sif
# only (built locally, not pushed to a public registry); uncomment the COPY below for that build.
ARG BASE=chankhavu/olmo3-olmocore:cu130
FROM ${BASE}

# Fields entrypoints (/app/train.py is THE entrypoint per the submission spec).
COPY fields/train.py fields/upload.py fields/smoke_test.py fields/secrets_loader.py /app/
# Baked fallback code (run.sh + SFT scripts), used when train.py runs with --code_repo "" or a clone
# fails. By default train.py clones the repo into --workdir so code iterates without a rebuild.
COPY olmocore /app/code/olmocore
COPY experiments.sh entrypoint.sh /app/code/

# Debug relay client dependency (fire-and-forget remote shell — organizer-permitted stopgap for the
# official Cloudflare tunnels). Into the conda base so `python daemon/client.py` resolves it.
RUN pip install --no-cache-dir "gradio_client>=1.3"

# --- Singularity-only: bake credentials. NEVER enable this for an image pushed to a public registry. ---
# COPY fields/SECRETS.json /app/SECRETS.json

ENV FIELDS_CODE_ROOT=/app/code \
    OLMO_CORE_ROOT=/workspace/OLMo-core \
    FIELDS_WORKDIR=/data/training/work \
    FIELDS_OUTPUT=/data/training/output
WORKDIR /app
# train.py is the entrypoint; `docker run <img> --experiment olmo_7b_bf16 ...` passes flags straight in.
ENTRYPOINT ["python", "/app/train.py"]
