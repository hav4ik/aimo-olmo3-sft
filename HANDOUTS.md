# HANDOUTS — Olmo 3 SFT, deploy + data prep (for the next agent)

You're picking up an Olmo 3 SFT pipeline for an **AIMO** (AI Math Olympiad) effort. Code + images
are built and committed; your two jobs are **(1) trigger the Docker images on Vast.AI / GPU clusters
and debug them live**, and **(2) prepare the REAL training dataset** (everything so far runs on a
throwaway *smoke* dataset — see §6). Read this once top-to-bottom.

---

## 0. Status in one breath
- Two frameworks: **OLMo-core** (reference — Olmo was trained with it) and **Axolotl** (fallback).
- Two precision arms each: **BF16** (default) and **FP8** (opt-in; §5).
- A deployable **mono-repo** (this repo, branch `olmo3-sft`) + two **DockerHub images** are built.
  Confirm what's pushed before relying on it: `git log origin/olmo3-sft -1` and
  `docker manifest inspect chankhavu/olmo3-olmocore:cu130`. Vast.AI runs fail at the runtime
  `git clone` / image pull until the push happens.
- All runs so far use the SMOKE dataset `allenai/tulu-3-sft-personas-math`. The real `<think>` math
  dataset is **your job** (§6) — there's now ONE offline prep that feeds both frameworks.

---

## 1. Map — repos & images
**Workspace:** `/home/vu/Workspace/AIMO-proof-pilot/sft-images/` — three dirs, no legacy:

| Dir | What | Remote |
|---|---|---|
| `aimo-olmo3-sft/` | **THE repo** (this one, branch `olmo3-sft`): `olmocore/` · `axolotl/` · `data_prep/` · `docker/` · `entrypoint.sh` · `bootstrap.sh` | → `github.com/hav4ik/aimo-olmo3-sft` |
| `OLMo-core/` (fork, branch `olmo3-sft`) | AI2's repo; only edit = FA2 `+sm120` in `src/Dockerfile`. Builds the OLMo-core base image | local |
| `open-instruct/` (branch `olmo3-sft-dataprep`) | AI2's data tooling + `Dockerfile.dataprep` (builds the dataprep image) | local |

**Docker images** (`docker images`):
| Image | Role |
|---|---|
| `chankhavu/olmo3-olmocore:cu130` (12 GB) | **deploy** — base + bootstrap (clones this repo at runtime) |
| `chankhavu/olmo3-axolotl:cu130` (19 GB) | **deploy** |
| `olmo-core-sft:cu130` | base: AI2 deps + olmo_core + FA2(sm90;100;120)+FA3+FA4 (built from the OLMo-core fork) |
| `axolotl-olmo3-sft:0.1.0` | base: axolotl-uv cu130 + FA3 wheel |
| `open-instruct-dataprep:0.1.0` (2.3 GB) | **data prep** — tokenizes SFT data → OLMo-core `.npy` |

**Runtime-clone design:** deploy images bake only `bootstrap.sh` (ENTRYPOINT) → at start it
`git clone`s `hav4ik/aimo-olmo3-sft@${CODE_BRANCH:-olmo3-sft}` into `/workspace/code` and execs
`entrypoint.sh`. So iterate on configs/scripts by **`git push`, no image rebuild**. No code, secrets,
or data are baked into the images.

---

## 2. Run on Vast.AI — env contract
Mount a persistent volume at **`/data/training`** (pre-populated with prepped data — §6), pass
secrets as **`-e`**. `entrypoint.sh` dispatches on env:

| Var | Meaning |
|---|---|
| `FRAMEWORK` | `olmocore` \| `axolotl` (unset ⇒ **shell** — your debug entry) |
| `PRECISION` | `bf16` (default) \| `fp8` |
| `MODEL_SIZE` | axolotl only: `7b` (default) \| `32b` → picks `configs/olmo3-<size>-<precision>.yaml` |
| `STAGE` | olmocore only: `train` (default, auto-converts HF→distcp on first run) \| `convert` (just produce the checkpoint & exit) |
| `DATASET_NAME` | which prepped dataset under `/data/training/datasets/<NAME>/` to train on |
| `NPROC_PER_NODE` | GPUs/node (default = all visible) |
| `NNODES`, `NODE_RANK`, `HEAD_NODE_IP`, `NCCL_PORT` | multi-node (e.g. 16×H200); single-node otherwise |
| `HF_TOKEN`, `WANDB_API_KEY` | secrets via `-e`; no WANDB key ⇒ offline mode |
| `SEQ_LEN`/`SEQUENCE_LEN`, `GLOBAL_BATCH_SIZE`, `MAX_STEPS`, `LR`, `EPOCHS` | recipe overrides — **small values for a single-GPU smoke** |
| `OLMO_ATTN_BACKEND`/`ATTN_IMPL`, `OLMO_FP8`, `CODE_BRANCH` | force attention / FP8 recipe / code branch |

**Attention auto-selects by GPU arch:** sm_90 (H100/H200) → `flash_3`/`flash_attention_3`; sm_120
(RTX PRO 6000 / Blackwell) & others → `flash_2` (FA2 built with sm_120) / `flex_attention`.

```bash
# Axolotl BF16 smoke on 1 RTX 6000 (data must be prepped first — §6):
docker run --rm --gpus all -v /data/training:/data/training \
  -e HF_TOKEN=$HF_TOKEN -e WANDB_API_KEY=$WANDB_API_KEY \
  -e FRAMEWORK=axolotl -e DATASET_NAME=mymath -e PRECISION=bf16 -e SEQUENCE_LEN=2048 -e MAX_STEPS=10 \
  chankhavu/olmo3-axolotl:cu130

# OLMo-core: just train — HF→distcp convert runs automatically on first use
# (one-time per /data/training volume, CPU; cached after). Add -e STAGE=convert to
# pre-stage the checkpoint without training (e.g. once before a multi-node job).
docker run --rm --gpus all -v /data/training:/data/training \
  -e HF_TOKEN=$HF_TOKEN -e WANDB_API_KEY=$WANDB_API_KEY \
  -e FRAMEWORK=olmocore -e DATASET_NAME=mymath -e PRECISION=bf16 \
  -e SEQ_LEN=2048 -e GLOBAL_BATCH_SIZE=4096 -e MAX_STEPS=10 chankhavu/olmo3-olmocore:cu130
```
Full recipe (8×H200): drop the smoke overrides (defaults = seq 32768 / 1,048,576 tok / 2 epochs).

---

## 3. Live-debugging playbook
- **Shell in:** run with **no `FRAMEWORK`** → bash in the cloned code at `/workspace/code`. Run
  `bash olmocore/run.sh` / `bash axolotl/run.sh` by hand, edit, re-run. `run.sh` is the single runner
  (single + multi-node via `NNODES`).
- **GPU/arch:** `nvidia-smi --query-gpu=name,compute_cap --format=csv` — `9.0`=Hopper, `12.0`=Blackwell.
- **Gotchas:**
  - **TE / `libnvrtc`:** on some nodes `import transformer_engine` dies on `ldconfig -p | grep libnvrtc`.
    In-container fix: `echo $(dirname $(find /opt/conda -name 'libnvrtc.so*'|head -1)) >/etc/ld.so.conf.d/zz.conf && ldconfig`.
  - `Supported flash-attn versions … 2.8.2` — harmless TE warning.
  - **7B full-FT memory:** fp32 AdamW ≈ **112 GB** — does NOT fit one 96 GB RTX 6000. Use **≥2 GPUs**
    (FSDP shards it) for the full recipe; single card = small-shape smoke only.
  - **CP at seq 32768** auto-engages (`cp_degree=2`) → needs **≥2 GPUs**; single-GPU smokes use `SEQ_LEN=2048`.
  - **FP8 + flex_attention on Blackwell is UNTESTED** — first RTX 6000 run = **BF16**; FP8 is proven on
    Hopper (`flash_attention_3`/`flash_3`). See `STABILITY.md`.

---

## 4. Models & recipe (canonical — don't drift)
- **7B = `allenai/Olmo-3-7B-Think`** (a *Think* checkpoint we **continue-tune**, not a base).
  **32B = `allenai/Olmo-3.1-32B-Think`**. (No Olmo-3.1 7B Think exists; canonical base if ever needed
  is `Olmo-3-1025-7B`.)
- **Arch:** `Olmo3ForCausalLM`, 32 layers, **3-sliding : 1-full** attention (window 4096 → full at
  `[3,7,11,15,19,23,27,31]`), dolma2 tokenizer **vocab 100278**, YaRN, **two eos** `<|im_end|>`(100265)
  + `<|endoftext|>`(100257).
- **Recipe = AI2 `7b_think_sft`:** lr **5e-5**, **2 epochs**, global batch **1,048,576 tok**, seq
  **32768**, SkipStepAdamW, hsdp, selective AC, compile, bf16.
- **OLMo-core trainer** `olmocore/sft_scripts/Olmo-3-7B-SFT-local.py` = a beaker-stubbed copy of AI2's
  script; **a no-env run == AI2 bit-for-bit.**
- ⚠ **Continue-tune caveat:** lr 5e-5/2ep is tuned for SFT-from-*base*; on a Think model it can dilute
  the reasoning — watch eval, lower LR if needed.

---

## 5. Chat template & precision
- **Template:** Axolotl uses **`chat_template: tokenizer_default`** → the model's own think-aware
  `chat_template.jinja`. OLMo-core tokenizes via `data_prep/prepare.sh --template`, default
  **`olmo_thinker`** (the canonical think-SFT template). **Both are think-aware now.** If the real
  dataset has NO `<think>` traces, use `--template olmo_thinker_no_think_sft_tokenization` instead.
- **Precision:** default **BF16** (matches AI2 + Nemotron + DeepSeek — all post-train bf16). FP8 is
  opt-in. **OLMo-core FP8** = `OLMO_FP8=rowwise` (default; per-channel — DeepSeek/Unsloth favor
  fine-grained) keeping **all attention + lm_head + embeddings BF16** (`fp8_attention_ignores`, 128
  FQNs; DeepSeek-V3 recipe). **Axolotl FP8** = accelerate `tensorwise` only (no granularity control),
  requires `torch_compile`. Validate loss parity vs bf16 first. Details: `STABILITY.md`, `RECIPES.md`.

---

## 6. DATA PREP & CONVERSION — your main job
The current dataset is a **SMOKE** (`allenai/tulu-3-sft-personas-math`) — it only proves the code path.
The real run uses a **`<think>`-formatted math dataset** (for AIMO). **One offline prep feeds both
frameworks** and accepts any input form.

**Where it runs:** on a box with docker + `open-instruct-dataprep:0.1.0` (the dev box — NOT inside a
Vast.AI training container). It writes `/data/training/datasets/<NAME>/`; you then mount
`/data/training` on the training node.

### `data_prep/prepare.sh --name <NAME> --input <SPEC> [opts]`
Produces, under `/data/training/datasets/<NAME>/`:
- `messages.parquet` — normalized `[{role,content}]`, **shared** (Axolotl reads it directly)
- `olmocore/` — tokenized `.npy` for OLMo-core (open-instruct + the chat template)

Two phases: (1) `normalize.py` resolves the input → one `messages` parquet (so **both frameworks see
byte-identical examples**); (2) open-instruct tokenizes that parquet → `.npy`.

**`--input <SPEC>` accepts (repeatable):**
- HF dataset id: `org/name`
- parquet files **inside** an HF repo: `--input org/name --data-files "data/train-*.parquet"`
- local parquet glob / comma-list: `"/data/raw/*.parquet"` or `a.parquet,b.parquet`
- `hf://datasets/org/name/*.parquet`

**Other opts:** `--template olmo_thinker` (think data) | `olmo_thinker_no_think_sft_tokenization`
(non-think) ; `--max-seq 32768` ; `--max-examples N` ; `--prompt-field/--response-field/--system-field`
if the source has no `messages` column.

```bash
./data_prep/prepare.sh --name mymath --input allenai/tulu-3-sft-personas-math --template olmo_thinker
./data_prep/prepare.sh --name mymath --input org/reasoning --data-files "data/train-*.parquet"
./data_prep/prepare.sh --name mymath --input "/data/raw/*.parquet" --max-seq 32768
./data_prep/prepare.sh --name mymath --input org/qa --prompt-field question --response-field answer
```
Then train with **`-e DATASET_NAME=mymath`** (both frameworks read the right artifacts automatically).

### Checkpoint conversion (OLMo-core only — automatic)
OLMo-core can't load HF weights directly, so the HF→distcp convert is required — but it now runs
**automatically on the first olmocore run** (one-time per `/data/training` volume, CPU; a
`.convert_complete` sentinel marks it done and every later run reuses it). Output lands in
`/data/training/checkpoints/olmocore-olmo3-7b-think`. To pre-stage it without training (e.g. once
before a multi-node job, so only NODE_RANK=0 converts), run the image with `-e STAGE=convert`.

### Notes
- `normalize.py` validated end-to-end (HF → parquet). open-instruct's loader natively accepts the
  local parquet (it branches on the `.parquet` extension), so the tokenize step just points at it.
- Keep the **same `--name`/dataset on both frameworks** so the axolotl-vs-OLMo-core comparison is honest.
- `--template` must match the data's think-format (see §5). `--visualize True` prints a tokenized+masked
  sample during prep — **eyeball it** before a long run. Masking/format audit: `DATA.md`.

---

## 7. Open issues / TODOs
1. **Push** the repo (`git push`) + the two deploy images (`docker push`) — until then Vast.AI fails at
   clone/pull. The dataprep image (`open-instruct-dataprep:0.1.0`) isn't on DockerHub either — run prep
   on the dev box, or push it as `chankhavu/olmo3-dataprep`.
2. **Pick the OLMo-core `--template`** for the real data: `olmo_thinker` if it carries `<think>` traces,
   else `olmo_thinker_no_think_sft_tokenization`.
3. **FP8 on Blackwell (RTX 6000)** untested (esp. FP8 + `flex_attention`) — first RTX 6000 run = BF16.
4. **Exported `generation_config.eos_token_id`** should list **both** `[100265, 100257]` (OLMo-core's HF
   converter does this; for axolotl set at export/serving).
5. **32B is HELD until the 7B is validated — see `SCALEUP_32B.md`** for the full plan. Axolotl 32B
   is ready (`olmo3-32b-{bf16,fp8}.yaml`, `MODEL_SIZE=32b`) and recipe-accurate — note the **32B
   recipe differs from the 7B: lr 1e-4 + 4.19M-tok batch** (vs 5e-5 / 1.05M). The OLMo-core 32B
   *reference* path is NOT wired yet (trainer `Olmo-3-7B-SFT-local.py` is 7B-only); the held work
   (stub AI2's `Olmo-3-32B-SFT.py` + a `MODEL_SIZE`/arch switch in olmocore/run.sh) is in SCALEUP_32B.md.

---

## 8. Where to look
- `README.md` — deploy quickstart + env table.
- `data_prep/` — `prepare.sh` (orchestrator), `normalize.py` (input → messages parquet), `convert_hf_to_olmocore.sh`.
- `RECIPES.md` — Axolotl vs OLMo-core recipe table · `STABILITY.md` — BF16-vs-FP8 test + FP8 internals.
- `SCALEUP_32B.md` — the 32B plan (held until the 7B is validated; recipe differs, OLMo-core path to wire).
- `DATA.md` — token-level chat-template / masking audit (read before changing data format).
