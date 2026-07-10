#!/usr/bin/env python3
"""Single-command CLI front-end for the olmo-core SFT container (in-container `docker run` style).

Turns `--flags` into the environment the runtime chain consumes
(bootstrap.sh -> entrypoint.sh -> run.sh -> torchrun -> the SFT script) and then execs the baked
bootstrap. Anything you don't pass falls back to run.sh's per-model defaults.

Run it AS the docker command (no host-side launcher needed):

    docker run --rm --gpus all --ipc=host \\
        -e HF_TOKEN=$HF_TOKEN -v /host/data:/data/training \\
        chankhavu/olmo3-olmocore:cu128-fa2-sink \\
        python /usr/local/bin/train.py \\
            --dataset chankhavu/yccchen-stage2-olmocore-256k-v2 \\
            --seq-len 65536 --epochs 2 --ac-budget 0.8 --cp-style ulysses

128-token smoke:
    ... python /usr/local/bin/train.py --seq-len 128 --max-steps 10 --gbs 1024 \\
        --max-tokens-per-rank 128 --ac-budget 0.8

Use --dry-run to print the resolved env without launching. This is the in-container twin of the
host-side olmocore/launch.sh (same flags/defaults); use whichever fits how you start the container.
"""
import argparse
import os
import sys

# (flag_dest, env_var, help). Defaults mirror olmocore/launch.sh (the current yccchen sink run);
# unset flags are simply not exported, so run.sh applies its own per-model defaults.
_MAP = [
    ("data",               "DATA",                     "host dir mounted to /data/training"),
    ("code_ref",           "CODE_REF",                 "aimo-olmo3-sft branch/tag/sha cloned at runtime"),
    ("model",              "HF_MODEL",                 "HF base model"),
    ("dataset",            "DATASET_HF",               "HF tokenized dataset"),
    ("dataset_subdir",     "DATASET_SUBDIR",           "subdir with token_ids_part_*.npy ('' = repo root)"),
    ("dataset_name",       "DATASET_NAME",             "local dir name under /data/training/datasets"),
    ("model_size",         "MODEL_SIZE",               "7b|32b"),
    ("sft_script",         "SFT_SCRIPT_NAME",          "SFT script filename"),
    ("seq_len",            "SEQ_LEN",                  "sequence length"),
    ("epochs",             "EPOCHS",                   "number of epochs (ignored if --max-steps set)"),
    ("max_steps",          "MAX_STEPS",                "steps mode (overrides --epochs)"),
    ("gbs",                "GLOBAL_BATCH_SIZE",        "global batch size in tokens"),
    ("max_tokens_per_rank","OLMO_MAX_TOKENS_PER_RANK", "per-rank token cap that sets cp_degree"),
    ("cp_style",           "OLMO_CP_STYLE",            "ring|ulysses"),
    ("attn_backend",       "OLMO_ATTN_BACKEND",        "attention backend: flash_2 | flash_3 (FA3; Hopper sm_90 only). Both do exact sink post-correction. Unset = auto by GPU arch (flash_3 on Hopper, flash_2 elsewhere)"),
    ("lr",                 "LR",                       "PEAK learning rate (default 5e-5 for 32B)"),
    ("lr_alpha_f",         "OLMO_LR_ALPHA_F",          "LR FLOOR as a fraction of peak: floor = alpha_f * --lr (default 0.1 -> 5e-6 at peak 5e-5; 0 = decay to 0)"),
    ("warmup_fraction",    "OLMO_LR_WARMUP",           "LR warmup as a fraction of total training (default 0.03, i.e. 3 percent)"),
    ("ac_budget",          "OLMO_AC_BUDGET",           "activation-checkpointing: <0..1> budget (higher=faster/more mem), 'none', or unset=selected_modules"),
    ("persistent_reductions", "OLMO_PERSISTENT_REDUCTIONS", "torch.compile RMSNorm reduction: 1=persistent (fast, needs ~200KB smem: Hopper/B200), 0=looped (fits small smem: RTX 6000/A100); unset=auto-detect from GPU smem"),
    ("fused_rmsnorm",      "OLMO_FUSED_RMSNORM",       "wide RMSNorm impl: 1=FusedRMSNorm (flash triton kernel, required on RTX 6000/A100 — the compiled persistent reduction won't fit smem), 0=stock compiled rms; unset=auto-on when GPU smem<200KB"),
    ("fused_lce",          "OLMO_FUSED_LCE",           "Liger fused-linear cross-entropy: 1=DEFAULT (no materialized (T,vocab) logits, ~10GB+ saved), 0=materialized reference (A/B the loss)"),
    ("model_dtype",        "OLMO_MODEL_DTYPE",         "master-weight dtype: float32=DEFAULT (fp32 master) | bfloat16 (bf16 master, ~8GB/rank less, RISKY — no stochastic rounding in olmo-core, validate loss)"),
    ("optim",              "OLMO_OPTIM",               "optimizer: skip_step (DEFAULT; SkipStepAdamW spike protection + bf16 moments = ~16GB/rank less) | fused_adamw (fp32 fused-kernel baseline, fastest). NOTE: adamw8bit is UNSUPPORTED under FSDP2 (bitsandbytes has no DTensor support) and errors at config time"),
    ("optim_dtype",        "OLMO_OPTIM_DTYPE",         "Adam moment dtype for --optim skip_step: DEFAULT bf16 (m/v in bf16, ~16GB/rank less, fp32 master kept); set fp32 to force fp32 moments. Ignored by fused_adamw"),
    ("grad_reduce_dtype",  "OLMO_GRAD_REDUCE_DTYPE",   "FSDP gradient reduce-scatter dtype: bf16 halves the grad buffer + comm (~8GB/rank less, slight grad-sum rounding; fp32 master unaffected); default fp32"),
    ("nodes_per_fsdp_group","OLMO_NODES_PER_FSDP_GROUP","how many NODES form one FSDP shard group (one copy of the sharded model). 1=shard within a node (default); raise it to shard the 32B floor across more nodes (~1/N the floor for N nodes) at the cost of inter-node all-gather"),
    ("sink",               "OLMO_USE_SINK",            "per-head attention sink 0|1"),
    ("sink_init",          "OLMO_SINK_INIT",           "initial sink logit (stock warm start only)"),
    ("hf_tokenizer",       "OLMO_HF_TOKENIZER",        "1=reuse model tokenizer, or an HF id"),
    ("stage",              "STAGE",                    "train|convert"),
    ("keep_ckpts",         "OLMO_KEEP_LAST_CKPTS",     "max persistent checkpoints kept on disk, oldest deleted (default 3 for 32B ~= 1TB; 0=keep all)"),
    ("save_interval",      "OLMO_SAVE_INTERVAL",       "steps between PERSISTENT checkpoints (default 1000)"),
    ("ephemeral_interval", "OLMO_EPHEMERAL_INTERVAL",  "steps between EPHEMERAL (rotating resume) checkpoints; must be < save-interval (default 500)"),
    ("hf_upload_repo",     "OLMO_HF_UPLOAD_REPO",      "HF model repo id (e.g. user/olmo3-32b-sft-128k) — a node-0 watchdog converts+ships each new checkpoint to <repo>/step<N>/ and the final model to the repo root. Needs HF_TOKEN with WRITE scope. Unset = no upload"),
    ("hf_upload_interval", "OLMO_HF_UPLOAD_INTERVAL",  "seconds between upload-watchdog polls (default 300)"),
    ("self_check",         "OLMO_ATTN_SELFCHECK",      "1=run the attention-sink kernel self-check before training"),
    ("wandb_project",      "WANDB_PROJECT",            "Weights & Biases project name"),
    ("wandb_entity",       "WANDB_ENTITY",             "Weights & Biases entity/team"),
    ("nnodes",             "WORLD_SIZE",               "number of NODES"),
    ("node_rank",          "GLOBAL_RANK",              "this node's index"),
    ("master_addr",        "MASTER_ADDR",              "rendezvous host"),
    ("master_port",        "MASTER_PORT",              "rendezvous port"),
]

# Defaults: match launch.sh so the no-flag run trains the current model on its own tokenizer + sink,
# single 8-GPU node. Values left None are not exported (run.sh decides).
_DEFAULTS = {
    "data": "/data/training",
    "code_ref": "olmocore-cu128-fa2-sink",
    "model": "chankhavu/yccchen-olmo3-deploy",
    "dataset": "chankhavu/yccchen-stage2-olmocore-256k-v2",
    "dataset_subdir": "",
    "dataset_name": "yccchen-stage2",
    "model_size": "32b",
    "sft_script": "Olmo-3-32B-SFT-bf16.py",
    "sink": "1",
    "hf_tokenizer": "1",
    "epochs": "2",
    "nnodes": "1",
    "node_rank": "0",
    "master_addr": "127.0.0.1",
    "master_port": "29400",
}


def main():
    # On Beaker (multi-node), bootstrap.sh's shim maps BEAKER_REPLICA_* -> WORLD_SIZE/GLOBAL_RANK/
    # MASTER_ADDR. Drop train.py's single-node rendezvous DEFAULTS so they don't clobber that shim (every
    # replica would otherwise think it's a 1-node job at 127.0.0.1). An explicit --nnodes/--node-rank/...
    # still wins; non-Beaker single-node launches keep the defaults.
    if os.environ.get("BEAKER_REPLICA_COUNT"):
        for _k in ("nnodes", "node_rank", "master_addr", "master_port"):
            _DEFAULTS.pop(_k, None)
    p = argparse.ArgumentParser(
        description="Single-command trainer CLI for the olmo-core SFT container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for dest, env, helptxt in _MAP:
        p.add_argument("--" + dest.replace("_", "-"), dest=dest,
                       default=_DEFAULTS.get(dest), help=f"-> {env}: {helptxt}")
    p.add_argument("--dry-run", action="store_true", help="print the resolved env and exit")
    args = p.parse_args()

    env = os.environ.copy()
    # MAX_STEPS overrides EPOCHS (mirror launch.sh): drop EPOCHS when steps mode is requested.
    if args.max_steps is not None:
        args.epochs = None
    for dest, evar, _ in _MAP:
        val = getattr(args, dest)
        if val is not None and str(val) != "":
            env[evar] = str(val)
    # DATASET_SUBDIR must be exported even when empty ("" = dataset shards live at the repo root).
    env["DATASET_SUBDIR"] = "" if args.dataset_subdir is None else str(args.dataset_subdir)

    resolved = {evar: env[evar] for _, evar, _ in _MAP if evar in env}
    print("[train.py] resolved run env:", flush=True)
    for k in sorted(resolved):
        print(f"    {k}={resolved[k]}", flush=True)
    for secret in ("HF_TOKEN", "WANDB_API_KEY"):
        print(f"    {secret}={'set' if os.environ.get(secret) else 'UNSET'}", flush=True)

    if args.dry_run:
        print("[train.py] --dry-run: not launching.", flush=True)
        return

    bootstrap = "/usr/local/bin/bootstrap.sh"
    if not os.path.exists(bootstrap):
        sys.exit(f"[train.py] ERROR: {bootstrap} not found — is this the SFT container image?")
    os.execvpe("bash", ["bash", bootstrap], env)


if __name__ == "__main__":
    main()
