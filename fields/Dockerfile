# Fields Model deploy image — bakes the /app entrypoints; everything heavy (base model + tokenized
# dataset) is downloaded at RUNTIME into --workdir. Built as a thin layer on the olmocore deploy image
# (olmo_core source + FA2/FA3 + TransformerEngine + the libnvrtc fix + the ring-flash-attn shim).
#
# Build context = repo root:
#   docker build -f fields/Dockerfile -t chankhavu/olmo3-fields:cu130 .
#
# CREDENTIALS (disclosed): if `fields/SECRETS.json` is present at build time, it is baked to
# /app/SECRETS.json (HF_TOKEN + WANDB_API_KEY) so the container needs no `-e`. If it is absent, only
# the `.example` template is copied (ignored at runtime) and credentials come from the environment.
# secrets_loader.py loads /app/SECRETS.json; a same-named env var always wins. DO NOT push an image
# built WITH a real SECRETS.json to a PUBLIC registry — bake it only for the local Singularity .sif.
#
# DEBUG SHELL (disclosed): train.py launches a fire-and-forget relay client on every node by default
# (organizer-permitted stopgap for the official Cloudflare tunnels). Disable with --no-remote-shell.
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

# Ship a REAL, self-contained nvcc first on PATH so the container never resolves a HOST nvcc. On the
# NII cluster a host nvcc on a noexec mount got picked up via PATH and raised an uncaught
# PermissionError during the first torch.compile, killing every rank (torch tolerates a *missing* nvcc
# but not a non-executable one). We install the matched CUDA-13.0 compiler trio (nvcc + nvvm/cicc +
# crt) as pip wheels — the same channel as our cu13 runtime — and expose nvcc via a stable wrapper at
# /opt/fields/bin that execs the real binary by ABSOLUTE path. (A symlink/dir-symlink breaks nvcc's
# include-root detection — it keys off argv[0]; the exec-absolute wrapper preserves it.) Versions are
# pinned EQUAL so cicc's emitted PTX matches the sibling ptxas (a skew makes real compiles fail).
RUN pip install --no-cache-dir \
        nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88 && \
    mkdir -p /opt/fields/bin && \
    REAL="$(find /opt/conda -name nvcc -path '*cu13*' -type f | head -1)" && \
    printf '#!/bin/sh\nexec %s "$@"\n' "$REAL" > /opt/fields/bin/nvcc && \
    chmod 0755 /opt/fields/bin/nvcc && \
    /opt/fields/bin/nvcc --version           # build-time smoke test: fail the build if nvcc is broken

# A: prefer ALL our baked-in binaries over anything a host bind could inject — populated in-image dirs
# FIRST, the empty host-fillable dirs (/usr/local/{nvidia,cuda}/bin, /root/.cargo/bin) LAST. Generalizes
# the nvcc fix to the whole toolchain (gcc/as/ld/python/pip/torchrun/...). The empty dirs stay on PATH
# (harmless — --nv puts nvidia-smi in /usr/bin) but can no longer SHADOW an in-image binary.
ENV PATH=/opt/fields/bin:/opt/conda/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.cargo/bin
# C: pin the highest-impact compilers to ABSOLUTE in-image paths so a shadowed gcc/g++ can't hijack
# inductor's C++ wrapper compile or our nvcc's host-compile step (holds even under a PATH override).
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
ENV NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc"

# Bake credentials IF fields/SECRETS.json is supplied (else only the .example template, which is
# ignored at runtime). The glob always matches the .example, so the build never fails on a missing
# SECRETS.json. secrets_loader reads /app/SECRETS.json only; /app/SECRETS.example.json is inert.
COPY fields/SECRETS*.json /app/

# Defaults live UNDER /tmp — the dir the cluster always binds (e.g. `singularity run --containall
# --bind <host>:/tmp …`), so downloads/checkpoints land on a real volume even when nothing else is
# bound. Override with --workdir/--output (or FIELDS_WORKDIR/FIELDS_OUTPUT/HF_HOME).
ENV FIELDS_CODE_ROOT=/app/code \
    OLMO_CORE_ROOT=/workspace/OLMo-core \
    FIELDS_WORKDIR=/tmp/olmo-sft/work \
    FIELDS_OUTPUT=/tmp/olmo-sft/output \
    HF_HOME=/tmp/olmo-sft/hf_cache
WORKDIR /app
# train.py is the entrypoint; `docker run <img> --experiment olmo_7b_bf16 ...` passes flags straight in.
ENTRYPOINT ["python", "/app/train.py"]
