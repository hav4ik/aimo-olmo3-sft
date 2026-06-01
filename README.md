# aimo-olmo3-sft — Olmo 3 SFT, deployable on Vast.AI + H200

One mono-repo, two frameworks (**OLMo-core** = reference, **Axolotl** = fallback), two
precision arms each (**BF16** / **FP8**). The heavy deps live in prebuilt DockerHub images;
**this repo's code is pulled at container start** (runtime `git clone`), so iterating is just
`git push` — no image rebuild. **No secrets in the image**; tokens come from env, data from
`/data/training`.

```
olmocore/   sft_scripts/ (AI2's trainer, beaker-stubbed) · run.sh
axolotl/    configs/ (olmo3-7b-bf16|fp8.yaml) · run.sh
data_prep/  prepare.sh (offline prep, BOTH frameworks) · normalize.py · convert_hf_to_olmocore.sh
docker/     Dockerfile.olmocore · Dockerfile.axolotl · build_and_push.sh
bootstrap.sh  entrypoint.sh   HANDOUTS.md  RECIPES.md  STABILITY.md  DATA.md  SCALEUP_32B.md
```
**New here? Read `HANDOUTS.md`** — the full deploy + data-prep guide for picking this up.

## How it fits together
- **DockerHub images** (`hav4ik/olmo3-olmocore:cu130`, `hav4ik/olmo3-axolotl:cu130`) bake only
  `bootstrap.sh` as ENTRYPOINT. At start it clones `CODE_REPO@CODE_BRANCH` → `/workspace/code`
  and execs `entrypoint.sh`.
- **`entrypoint.sh`** dispatches on `FRAMEWORK` → `olmocore/run.sh` or `axolotl/run.sh`. No
  `FRAMEWORK` ⇒ shell.
- **Attention auto-selects by GPU arch**: sm_90 (H100/H200) → `flash_3` / `flash_attention_3`;
  sm_120 (RTX PRO 6000 / Blackwell) and others → `flash_2` (we built FA2 with sm_120) /
  `flex_attention`. So the same image runs on Vast.AI RTX 6000 and the H200 cluster.

## One-time setup
1. **Create** the GitHub repo `hav4ik/aimo-olmo3-sft` (public) and push this branch:
   ```bash
   git remote add origin https://github.com/hav4ik/aimo-olmo3-sft.git
   git push -u origin olmo3-sft
   ```
2. **Build + push the images** (needs your `docker login`):
   ```bash
   DOCKERHUB_USER=hav4ik ./docker/build_and_push.sh
   ```
   (The prebuilt base images `olmo-core-sft:cu130` / `axolotl-olmo3-sft:0.1.0` must exist on
   the build box — see the parent `sft-images` repo for how they're built.)

## Run (Vast.AI or H200)
**1. Prep data once** (on a box with docker — writes `/data/training/datasets/<NAME>/`; both
frameworks read it):
```bash
./data_prep/prepare.sh --name mymath --input allenai/tulu-3-sft-personas-math --template olmo_thinker
```
**2. Train** — mount `/data/training`, pass secrets + `DATASET_NAME`. Quick smoke on 1 RTX 6000:
```bash
# Axolotl BF16:
docker run --rm --gpus all -v /data/training:/data/training \
  -e HF_TOKEN=$HF_TOKEN -e WANDB_API_KEY=$WANDB_API_KEY \
  -e FRAMEWORK=axolotl -e DATASET_NAME=mymath -e PRECISION=bf16 -e SEQUENCE_LEN=2048 -e MAX_STEPS=10 \
  hav4ik/olmo3-axolotl:cu130

# OLMo-core: convert the checkpoint ONCE, then train:
docker run --rm --gpus all -v /data/training:/data/training -e HF_TOKEN=$HF_TOKEN \
  -e FRAMEWORK=olmocore -e STAGE=convert hav4ik/olmo3-olmocore:cu130
docker run --rm --gpus all -v /data/training:/data/training \
  -e HF_TOKEN=$HF_TOKEN -e WANDB_API_KEY=$WANDB_API_KEY \
  -e FRAMEWORK=olmocore -e DATASET_NAME=mymath -e PRECISION=bf16 \
  -e SEQ_LEN=2048 -e GLOBAL_BATCH_SIZE=4096 -e MAX_STEPS=10 hav4ik/olmo3-olmocore:cu130
```
Full recipe (8×H200): drop the smoke overrides (defaults seq 32768 / 1,048,576 tok / 2 epochs).
FP8: `-e PRECISION=fp8`. **Full env contract + data-prep guide in `HANDOUTS.md`.**

## Env contract
| Var | Meaning |
|---|---|
| `FRAMEWORK` | `olmocore` \| `axolotl` (unset ⇒ shell) |
| `PRECISION` | `bf16` (default) \| `fp8` |
| `MODEL_SIZE` | axolotl only: `7b` (default) \| `32b` → `configs/olmo3-<size>-<precision>.yaml` |
| `STAGE` | olmocore only: `train` (default) \| `convert` (HF→distcp checkpoint, run once) |
| `DATASET_NAME` | which prepped dataset under `/data/training/datasets/<NAME>/` to train on |
| `NPROC_PER_NODE` | GPUs/node (default = all visible) |
| `NNODES`, `NODE_RANK`, `HEAD_NODE_IP`, `NCCL_PORT` | multi-node (e.g. 16×H200); single-node otherwise |
| `HF_TOKEN`, `WANDB_API_KEY` | secrets, via `-e` — never baked. No WANDB key ⇒ offline mode |
| `SEQ_LEN`/`SEQUENCE_LEN`, `GLOBAL_BATCH_SIZE`, `MAX_STEPS`, `LR`, `EPOCHS` | recipe overrides (use small values for a single-GPU smoke) |
| `OLMO_ATTN_BACKEND` / `ATTN_IMPL`, `OLMO_FP8` | force attention / FP8 recipe instead of auto |
| `CODE_BRANCH` | which branch of this repo to pull (default `olmo3-sft`) |

## Data layout under `/data/training` (produced by `data_prep/prepare.sh --name <NAME>`)
- `datasets/<NAME>/messages.parquet` — normalized, shared (Axolotl reads this)
- `datasets/<NAME>/olmocore/` — tokenized `.npy` for OLMo-core
- `checkpoints/olmocore-olmo3-7b-think/` — converted base checkpoint (`STAGE=convert` makes it)
- `hf_cache/`, `wandb/` — created automatically
Both frameworks train the SAME prepped examples (selected by `DATASET_NAME`). See `HANDOUTS.md` §6.

## Notes
- **7B full-FT memory**: fp32 AdamW state ≈ 112 GB — tight on one 96 GB RTX 6000; use ≥2 GPUs
  (FSDP shards it) for the full recipe. Single-GPU is fine for a small-shape smoke.
- **CP at seq 32768** auto-engages (`cp_degree=2`) and needs ≥2 GPUs — another reason to smoke
  at `SEQ_LEN=2048` on a single card.
