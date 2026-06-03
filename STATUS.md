# STATUS — current state, requirements, open items (session handoff)

Branch `olmo3-sft`. Two deploy images: `chankhavu/olmo3-olmocore:cu130` (12 GB) and
`chankhavu/olmo3-axolotl:cu130` (19.8 GB). Both rebuilt locally this session; **not yet pushed**.
Several git commits **ahead of origin** — needs `git push origin olmo3-sft`.

## Launch interface (both frameworks)

One knob: **`EXPERIMENT=<size>_<precision>_<variant>`** (e.g. `7b_bf16_cot`, `32b_fp8_cot`).
`experiments.sh` resolves it → `MODEL_SIZE`, `PRECISION`, `DATASET_NAME`, `DATASET_HF`,
`DATASET_SUBDIR`, `RUN_NAME`. `entrypoint.sh` resolves EXPERIMENT then defaults `FRAMEWORK=olmocore`.
Cluster/topology env + secrets are the only other inputs. Code is **cloned at runtime** from
`github.com/hav4ik/aimo-olmo3-sft@olmo3-sft` (PUBLIC, anon HTTPS) into `/data/training/code`;
pin with `CODE_REF=<branch|tag|sha>`. So config/script changes ship on **`git push`, no rebuild**;
only Dockerfile changes need a rebuild+push.

## OLMo-core (reference engine)

- Flow: clone code → (rank-0) convert HF weights→distcp (`STAGE=convert` auto on first run, cached)
  → (rank-0) `hf download` pre-tokenized `.npy` (`DATASET_HF` + `DATASET_SUBDIR`) → `torchrun` train.
  Other nodes wait on sentinels (`THIS_NODE_RANK = NODE_RANK || GLOBAL_RANK || 0`).
- `MODEL_SIZE`: **7b** (`Olmo-3-7B-Think`, arch `olmo3_7b`, lr 5e-5, GBS 1,048,576) | **32b**
  (`Olmo-3.1-32B-Think`, `olmo3_32b`, lr 1e-4, GBS 4,194,304). **⚠ 32B training script
  `Olmo-3-32B-SFT-local.py` does NOT exist yet** — run.sh selects `Olmo-3-${MODEL_SIZE^^}-SFT-local.py`
  and fails cleanly if absent. So 32B converts + downloads data but **cannot train** until the script
  is created (beaker-stub the official `OLMo-core/src/scripts/train/sft/Olmo-3-32B-SFT.py`).
- **`SEQ_LEN` default = 65536** (model's full YaRN window 8192×8; data tokenized at 65536, 0 trunc).
- Multi-node: reads `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/GLOBAL_RANK` (set by cluster, never set by us);
  auto-detects process-level vs node-level `WORLD_SIZE/GLOBAL_RANK`; `NNODES/NODE_RANK` override.
  Wraps `torchrun` (static rendezvous when MASTER_ADDR set; c10d for Vast.AI multi-node; else standalone).
  One container per node; `--nproc_per_node` = local GPUs.
- Optimizer per precision: fused AdamW (bf16) / SkipStepAdamW (fp8) via `OLMO_OPTIM`. W&B on when
  `WANDB_API_KEY` set (entity/project env-driven; `WANDB_ENTITY`/`WANDB_PROJECT`).
- **Parallelism (VERIFIED):** OLMo-core flattens `cp` into the FSDP shard mesh
  (`get_dp_model_mesh` → `dp_cp = dp_shard × cp`). So params shard across **all** GPUs regardless of cp.
  On 4 GPUs at cp=4: ~28 GB/GPU params/optim + 16384-tok activations/rank → **65536 FITS on 4× RTX 6000**.
  Memory = sharding (FSDP for params, CP for activations); DP = throughput only (more DP ⇒ more
  activation, not less). cp_degree auto = ceil(SEQ_LEN / 16384) on H100/H200 (16384 = AI2's H100-tuned
  constant `MAX_RANK_MICROBATCH_SIZE_TOKENS`, NOT a hardware limit — 96 GB cards hold more).
  `OLMO_MAX_RANK_TOKENS` (cap → cp) knob was added then **removed** (default fits); re-add if cp
  tuning wanted (raise cap → fewer cp → more contiguous activation/device → spends spare VRAM).
- **Batch/microbatch tuning (NEW, see [BATCHING.md](BATCHING.md)):** `GLOBAL_BATCH_SIZE` sets global
  tokens/batch G; `RANK_MICROBATCH_TOKENS` sets per-DP-rank microbatch (per-**device** = ÷ cp; at
  65536 → cp=4). Identity `G = rmb × dp_world × grad_accum`; grad-accum auto-derived so G is fixed
  across any node count. E.g. `RANK_MICROBATCH_TOKENS=262144` → per-device 65536 (4×), grad-accum 4
  on 4 GPUs. Both knobs in `olmocore/run.sh`; off by default ⇒ no-env run is AI2 bit-for-bit.
- Image: torch 2.10.0+cu130, transformers 5.9.0, NCCL 2.28.9, FA2 sm90;100;120 + FA3 + FA4. nvrtc/TE
  linker fix **baked** (read-only-Singularity safe).
- **⚠ Checkpoint path wart (NOT fixed):** writes to
  `/data/training/checkpoints/checkpoints/<user|local>/olmo-sft/<run_name>/` (DOUBLE `checkpoints/`
  — `OLMO_SFT_SAVE_ROOT=$DATA/checkpoints` but the script appends `/checkpoints` again). Fix = set
  `OLMO_SFT_SAVE_ROOT=$DATA`. Persistent ckpt every 1000 steps (NOT pruned), ephemeral every 500.
  **distcp (sharded) format, NOT HF; nothing uploads it off the volume.**

## Axolotl (fallback engine)

- **Online data** (no 120 GB pre-stage): reads HF dataset directly (`DATASET_HF`/`DATASET_SRC`),
  tokenizes at launch (cached `/data/training/last_run_prepared`). Configs: `path: __DATASET__`
  (HF hub, no `ds_type`), `type: chat_template`, `field_messages: messages`.
- **Sequence parallelism for 65536 (BUILT + VERIFIED):** FA2 (grafted from olmo-core base, sm120) +
  `ring-flash-attn` + a transformers-5.x shim (`is_flash_attn_greater_or_equal_2_10` re-export) baked
  into `Dockerfile.axolotl` (build `COPY --from`s the olmocore image → build olmocore first).
  run.sh: `CONTEXT_PARALLEL_SIZE>1` → forces `flash_attention_2` + `--context_parallel_size`.
  axolotl SP needs `micro_batch_size=1` + `sample_packing` (configs already set). axolotl FSDP also
  flattens `dp_shard_cp`.
- attn auto: sm_90→FA3, sm_120→flex; CP forces FA2 (FA3 is Hopper-only; ring kernel is FA2-based).
- Configs still `sequence_len: 32768`; pass `SEQUENCE_LEN=65536` for long context. axolotl 0.16.2.dev0,
  transformers 5.8.1.

## Data

`chankhavu/smolmo-proofs-cot-sft` (HF, PUBLIC): `olmocore/` subfolder = pre-tokenized `.npy`
(token_ids/labels_mask, tokenized at max_seq **65536**, 0 skipped, ~20k mean / 65536 max tokens,
302k seqs, 6.08B tokens, 92.5% trainable); `data/` = 5 parquet shards (the messages, for axolotl,
~120 GB). variant `cot` → repo `chankhavu/smolmo-proofs-cot-sft`, subdir `olmocore`.

## VastAI run requirements (test box: 4× RTX 6000 Pro, Blackwell sm_120, 96 GB, PCIe)

```bash
# olmo-core @ 65536 (cp=4, fits ~28 GB/GPU). BF16 first.
docker run --rm --gpus all --ipc=host -v /workspace/data:/data/training \
  -e EXPERIMENT=7b_bf16_cot -e HF_TOKEN=hf_xxx -e WANDB_API_KEY=xxx \
  chankhavu/olmo3-olmocore:cu130
# axolotl @ 65536 (SP).
docker run --rm --gpus all --ipc=host -v /workspace/data:/data/training \
  -e FRAMEWORK=axolotl -e EXPERIMENT=7b_bf16_cot \
  -e SEQUENCE_LEN=65536 -e CONTEXT_PARALLEL_SIZE=2 \
  -e HF_TOKEN=hf_xxx -e WANDB_API_KEY=xxx chankhavu/olmo3-axolotl:cu130
```
- `--ipc=host` REQUIRED (dataloader/NCCL shm). ≥2 GPUs for 7B; 65536 uses all 4 (cp).
- ~50 GB+ free on the volume (convert ckpt + HF cache + data + ckpts).
- **Images must be `docker push`ed to `chankhavu/` first** (Vast.AI pulls them).
- axolotl smoke: `-e DATASET_SRC=<tiny hf set> -e MAX_STEPS=10 -e SEQUENCE_LEN=2048` to avoid 120 GB pull.

## Singularity / NII (ABCI) run requirements — DETAILED

**Hardware (ABCI compute node H):** 8× **H200 SXM 141 GB** (NVLink intra-node), 2× Xeon 8558 =
**96 physical cores** (192 logical), **2 TB RAM**, **~14 TB local NVMe** RAID0 at `$PBS_LOCALDIR`
(**WIPED at job end**), **InfiniBand NDR ×8 rails** inter-node. Drivers: **CUDA 13.0, driver
580.105.08, recommended NCCL 2.23.x** (we ship **2.28.9** — flagged, should be compatible).

**Runtime:** **SingularityCE 4.1/4.3/4.4** or SingularityPRO 4.1.12. **No Docker.** Scheduler =
**PBS Pro**; group `rt_HF` is node-exclusive (8 GPUs). 3-node job: `qsub -q rt_HF -l select=3:mpiprocs=192`.

**Build the SIF (from the pushed image):**
```bash
apptainer build olmo3-olmocore.sif docker://chankhavu/olmo3-olmocore:cu130
```

**Run (one container PER NODE; same command on every node):**
```bash
apptainer run --nv \
  --bind /path/to/run_storage:/data/training \
  --env EXPERIMENT=32b_fp8_cot,HF_TOKEN=$HF_TOKEN,WANDB_API_KEY=$WANDB_API_KEY \
  olmo3-olmocore.sif
```
- `run` (not `exec`) → honors the runscript (bootstrap). `--nv` → GPUs.
- **Apptainer inherits host env by default**, so the PBS job's `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/
  GLOBAL_RANK` are visible to the container automatically. Secrets via `--env` or `APPTAINERENV_*`.
- **No `--writable-tmpfs` needed**: nvrtc fix is baked; code clones into the bound (writable)
  `/data/training/code`; `/tmp` is writable by default.
- **`/data/training` bind MUST be writable and SHARED across all nodes** (rank-0 stages the converted
  ckpt + downloaded `.npy` there; other nodes read). Use a **persistent** tier (e.g. `/groups` Lustre),
  NOT `$PBS_LOCALDIR` (wiped at job end). The **"1 TB" = the working area** (model + data + checkpoints
  must fit) — confirm with NII.
- ABCI **S3** = `s3.v3.abci.ai` (region `ap-northeast-1`), reachable **only from compute/interactive
  nodes** (internal), creds via `~/.aws`. **External internet from compute nodes is UNCONFIRMED** —
  ABCI docs imply internal-only; the runtime code clone (GitHub) + HF model/data download assume
  outbound reachability. **THIS IS A LOAD-BEARING UNKNOWN.**

## Open questions sent / to send to organizers + NII

1. **Entrypoint contract (SENT):** our env-driven `apptainer run img` vs the `submissions-instructions`
   repo's `singularity run img python /app/train.py --model_path … --output_path …` CLI. Mutually
   exclusive — must confirm. If CLI, we add a `/app/train.py` wrapper.
2. **Multi-node (SENT):** which of MASTER_ADDR/PORT/WORLD_SIZE/GLOBAL_RANK they set, process- vs
   node-level semantics, one container per node (our model) vs per GPU.
3. **Result extraction (DRAFTED):** do they mount object storage (e.g. ABCI S3) or expect us to own
   auth/upload (our S3 presigned URL / HF)? Is the upload host air-gapped from external internet?
   Their repo says we provide `upload.py --s3_url --source_dir`, host-run after the container exits.
   Our checkpoints are **distcp (sharded)** — likely need distcp→HF consolidation before upload, and
   in multi-node the shards land on the shared storage (one upload from rank-0 sees all).
4. **Runtime internet (NOT yet asked, CRITICAL):** can compute nodes reach GitHub/HF/W&B at runtime?
   If not, bake code + pre-stage weights/data instead.

## Pending work (priority order)

1. **olmo-core 32B training script** `Olmo-3-32B-SFT-local.py` (beaker-stub official 32B script +
   our knobs: OLMO_ATTN_BACKEND, OLMO_FP8, OLMO_OPTIM, env-gated W&B). 32B blocked on this.
2. **Fix checkpoint double-`checkpoints/` wart** (`OLMO_SFT_SAVE_ROOT=$DATA`).
3. **Checkpoint retention vs 1 TB** (32B ckpt ~450 GB, persistent not pruned → trim).
4. **distcp→HF consolidation + `upload.py`** (pending organizer answer on #3 above).
5. **Push images** (`chankhavu/olmo3-{olmocore,axolotl}:cu130`) + **`git push`** (commits ahead).
6. Adversarial **review pass** of the multi-node/staging/rank-0 logic before NII handoff (no cluster iteration).
7. Optional: axolotl configs `sequence_len: 65536`; re-add `OLMO_MAX_RANK_TOKENS`.

## Decided / settled this session

- 65536 tokens/sample non-negotiable, both frameworks. Achieved (olmo-core cp=4 fits 4 GPUs;
  axolotl via SP). BF16 first; FP8 deferred (not removed). No framework-internal modifications.
- No TP (FSDP+CP suffices for 7B and 32B at this scale; TP only for models too big for FSDP+CP,
  and needs NVLink). 3D TP×CP×dp_shard is possible (TP+CP intra-node, dp_shard inter-node) but
  unnecessary here.
- Data: pre-tokenized .npy hosted on HF, pulled at runtime (olmo-core); axolotl tokenizes online.
- Single code source = the git ref (clone at runtime to /data/training/code); no baked code.
