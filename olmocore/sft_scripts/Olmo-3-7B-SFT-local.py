"""
LOCAL COPY of OLMo-core's `src/scripts/train/sft/Olmo-3-7B-SFT.py` (v2.5.0),
near-verbatim so we run AI2's ACTUAL recipe, not a reimplementation. The body
(BatchSizeConfig, model/optim/trainer config, train()) is unchanged; only the
Beaker/cluster coupling is stubbed so `train` runs under a plain `torchrun`
off-cluster. Re-diff against upstream when bumping the OLMo-core pin.

DIFFERENCES FROM UPSTREAM (minimal; a NO-ENV run == AI2's recipe bit-for-bit):
  1. Beaker decoupling — `olmo_core.internal.common` (CLUSTER_TO_GPU_TYPE,
     build_launch_config, get_beaker_username, get_root_dir, get_work_dir) and
     `olmo_core.launch.beaker.BeakerLaunchConfig` are replaced by local stubs
     below. Upstream's module-level `import beaker` makes the script unimportable
     without a Beaker cluster (even for `train`). get_root_dir() returns
     $OLMO_SFT_SAVE_ROOT (no Weka/Beaker); the `launch` subcommand is disabled
     (we only ever run `train` under a plain torchrun).
  2. GPUS_PER_NODE: hardcoded 8 -> taken from torchrun's LOCAL_WORLD_SIZE, so the
     1/2/8/16-GPU shapes compute the right world_size/shard_degree.
  3. Opt-in performance / escape-hatch env knobs, ALL OFF by default (so a no-env run
     IS AI2's recipe). The Hopper launcher turns the perf ones on for the H200 runs:
       - OLMO_ATTN_BACKEND=flash_3  -> FA3 packed-varlen (vs AI2 flash_2)
       - OLMO_FP8=rowwise|tensorwise|rowwise_with_gw_hp -> torchao float8 on the linears
       - OLMO_MODEL_DTYPE           -> override model storage dtype (else factory)

NOTHING ELSE DEVIATES — optimizer (SkipStepAdamW), hsdp, selective AC (feed_forward),
compile_model=True, generate_doc_lengths=True, YaRN are all EXACTLY upstream.
Context parallelism is AI2's own logic, untouched: BatchSizeConfig auto-sets
cp_degree (=2 at seq 32768 on H100/H200; cap is 16384 tok/rank, doubled on B200).
To run AI2's 7b_think_sft recipe, pass the SAME CLI overrides their open-instruct
script does: `--train_module.optim.lr=5e-5`, `--trainer.max_duration.value=2
--trainer.max_duration.unit=epochs`, `--global_batch_size=1048576`,
`--seq_len=32768` (see olmocore/run.sh).
"""

import argparse
import fnmatch
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast
from urllib.parse import urlparse

from rich import print

from olmo_core.config import Config, DType
from olmo_core.float8 import AOFloat8LinearRecipe, Float8Config  # DIFF #3: opt-in FP8
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.exceptions import OLMoConfigurationError
# --- LOCAL STUBS replacing olmo_core.internal.common (whose top-level `import
#     beaker` makes the script unimportable off-cluster). See DIFF #1 in the header.
CLUSTER_TO_GPU_TYPE = {  # gpu_type only gates microbatch caps; H100/H200 == "h100"
    "local_h100": "h100",
    "local_h200": "h100",
}


def get_root_dir(cluster: str) -> str:
    return os.environ.get("OLMO_SFT_SAVE_ROOT", "/data/checkpoints")


def get_work_dir(root_dir) -> str:
    return os.path.join(str(root_dir), "_work")


def get_beaker_username() -> str:
    return os.environ.get("USER", "local")


def build_launch_config(**kwargs):
    return None  # no Beaker/Gantry launch locally; the `launch` subcommand is disabled
# --- end local stubs ---
from olmo_core.io import copy_dir, dir_is_empty, get_parent, join_path, list_directory
from olmo_core.nn.attention import AttentionBackendName  # DIFF #4 (env attn backend)
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 16_384
DEFAULT_NUM_NODES = 1
# DIFF #2: upstream hardcodes 8; take it from torchrun (LOCAL_WORLD_SIZE = nproc
# per node) so the 1/2/8/16-GPU shapes compute the right world_size/shard_degree.
GPUS_PER_NODE = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
# Max tokens one rank holds per micro-batch (activation budget); drives cp_degree + grad-accum.
# AI2's 16384 is tuned for an 80GB H100; default keeps a no-env run AI2-exact. Override with
# OLMO_MAX_RANK_TOKENS on bigger cards (e.g. 96GB RTX 6000) to use more VRAM and cut cp_degree, so a
# longer SEQ_LEN fits on fewer GPUs (e.g. 32768 -> 65536 needs only cp=2 instead of 4). Raise it
# CAUTIOUSLY — it's an activation budget; set it past what the forward/backward fits and you OOM.
MAX_RANK_MICROBATCH_SIZE_TOKENS = int(os.environ.get("OLMO_MAX_RANK_TOKENS", "16384"))


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int  # assumes all ranks are either data parallel or context parallel
    gpu_type: str
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: Optional[int] = None

    def __post_init__(self):
        assert self.global_batch_size_tokens > 0, "global_batch_size_tokens must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert (
            self.sequence_length & (self.sequence_length - 1)
        ) == 0, "sequence_length must be a power of 2"
        assert self.world_size > 0, "world_size must be positive"
        assert (self.world_size & (self.world_size - 1)) == 0, "world_size must be a power of 2"

        # Determine max tokens per rank based on GPU type
        max_tokens_per_rank = MAX_RANK_MICROBATCH_SIZE_TOKENS
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        # Check if we need context parallelism based on sequence length
        if self.sequence_length > max_tokens_per_rank:
            # Calculate minimum CP degree needed to fit sequence length
            min_cp_degree = 2
            while (self.sequence_length // min_cp_degree) > max_tokens_per_rank:
                min_cp_degree *= 2

            self.cp_degree = min_cp_degree
            log.info(
                f"Sequence length ({self.sequence_length} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). Setting cp_degree={self.cp_degree}"
            )

        # Calculate rank batch size and grad accum steps
        cp_factor = self.cp_degree if self.cp_degree is not None else 1
        dp_world_size = self.world_size // cp_factor
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        # Ensure rank_batch_size_tokens doesn't exceed max_tokens_per_rank (adjusted by the cp_factor)
        if rank_batch_size_tokens > max_tokens_per_rank * cp_factor:
            # Need gradient accumulation
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > (
                max_tokens_per_rank * cp_factor
            ):
                self.grad_accum_steps *= 2

            self.rank_microbatch_size_tokens = rank_batch_size_tokens // self.grad_accum_steps
            log.info(
                f"Rank batch size ({rank_batch_size_tokens} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). "
                f"Using grad_accum_steps={self.grad_accum_steps}"
            )
        else:
            self.rank_microbatch_size_tokens = rank_batch_size_tokens
            self.grad_accum_steps = 1

        # Validate that rank_microbatch_size_tokens is divisible by sequence_length
        assert self.rank_microbatch_size_tokens % self.sequence_length == 0, (
            "rank_microbatch_size_tokens must be divisible by sequence_length (got "
            f"{self.rank_microbatch_size_tokens} and {self.sequence_length})"
        )
        self.rank_microbatch_size_sequences = (
            self.rank_microbatch_size_tokens // self.sequence_length
        )

        # Final validation
        total_tokens = self.rank_microbatch_size_tokens * dp_world_size * self.grad_accum_steps
        assert self.global_batch_size_tokens == total_tokens, (
            "global_batch_size_tokens must equal "
            "(rank_microbatch_size_tokens * dp_world_size * grad_accum_steps) (got "
            f"{self.global_batch_size_tokens} and {total_tokens})"
        )


def _separate_prefix_and_glob(prefix: str) -> Tuple[str, str]:
    if any(char in prefix for char in ["*", "?", "[", "]"]):
        parts = prefix.split("/")
        base_parts = []
        for part in parts:
            if any(char in part for char in ["*", "?", "[", "]"]):
                break
            base_parts.append(part)
    else:
        base_parts = prefix.split("/")
    if not base_parts:
        return ".", prefix

    new_prefix = "/".join(base_parts)
    glob_str = prefix[len(new_prefix) :]

    return new_prefix, glob_str.lstrip("/")


def glob_remote_dataset(prefix: str) -> List[str]:
    parsed_path = urlparse(prefix)
    scheme, bucket, parsed_prefix = (
        parsed_path.scheme,
        parsed_path.netloc,
        parsed_path.path.lstrip("/"),
    )
    parsed_prefix_pre_glob, glob_str = _separate_prefix_and_glob(parsed_prefix)
    base_prefix_without_scheme = Path(f"{bucket}/{parsed_prefix_pre_glob}")

    paths: List[str] = []

    for path in list_directory(f"{scheme}://{base_prefix_without_scheme}"):
        parsed_path = urlparse(path)
        path_without_scheme = Path(parsed_path.netloc) / parsed_path.path.lstrip("/")
        relative_to_base_prefix = path_without_scheme.relative_to(base_prefix_without_scheme)

        if glob_str and not fnmatch.fnmatch(str(relative_to_base_prefix), glob_str):
            # must match the glob if a glob was provided
            continue

        # path was valid!
        paths.append(path)

    return paths


def build_sft_dataset(
    root_dir: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    dataset_path: str,
) -> NumpyPackedFSLDatasetConfig:
    clean_path = dataset_path.rstrip("/")
    token_id_paths = [f"{clean_path}/token_ids_part_*.npy"]
    label_mask_paths = [f"{clean_path}/labels_mask_*.npy"]
    expand_glob = True

    dataset = NumpyPackedFSLDatasetConfig(
        # general config
        tokenizer=tokenizer_config,
        work_dir=get_work_dir(root_dir),
        paths=token_id_paths,
        expand_glob=expand_glob,
        label_mask_paths=label_mask_paths,
        generate_doc_lengths=True,  # ...and mask attention so that they don't attend to each other
        long_doc_strategy=LongDocStrategy.truncate,  # truncate docs...
        sequence_length=sequence_length,  # ...that are over this length
    )

    return dataset


def fp8_attention_ignores(model_config) -> List[str]:
    """FQNs of ALL attention q/k/v/o projections — kept OUT of FP8 to follow the
    DeepSeek-V3 recipe, which keeps every attention operator in BF16/FP32 (attention has
    the widest dynamic range; §3.3.1 lists "attention operators" among the components held
    at original precision). With these excluded, FP8 covers the feed-forward linears only.
    `lm_head.w_out` is already auto-excluded by Transformer.apply_fp8, and embeddings +
    norms are not nn.Linear (torchao float8 only touches nn.Linear), so none need listing.
    Computed from a meta build (no real allocation) so the exact FQNs track the model.
    Only used when OLMO_FP8 is set (FP8 is opt-in, off by default)."""
    meta = model_config.build(init_device="meta")
    ignores: List[str] = []
    for name, module in meta.named_modules():
        if name.endswith(".attention") and name.count(".") == 2:  # blocks.<i>.attention
            ignores.extend(f"{name}.{proj}" for proj in ("w_q", "w_k", "w_v", "w_out"))
    return ignores


@dataclass
class SFTConfig(Config):
    """
    Custom config class for the sft run.

    Making config classes isn't strictly necessary for OLMo-core, but it gives us a nice way to
    capture all of the hyperparameters for a run and an easy way to override those options from
    the command line without configuring a complicated command line parser.
    """

    run_name: str

    launch: Optional[Any]  # DIFF #1: was BeakerLaunchConfig; stubbed to None locally
    model: TransformerConfig
    dataset: Optional[NumpyPackedFSLDatasetConfig]
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        seq_len: int,
        num_nodes: int,
        global_batch_size: int,
        checkpoint: str,
        cluster: str,
        overrides: List[str],
        workspace: str,
        budget: str,
        init_seed: int = 33333,
        dataset_path: str,
    ) -> "SFTConfig":
        root_dir = get_root_dir(cluster)
        user_name = get_beaker_username()

        tokenizer_config = TokenizerConfig.dolma2()
        dataset_config = build_sft_dataset(
            root_dir=root_dir,
            tokenizer_config=tokenizer_config,
            sequence_length=seq_len,
            dataset_path=dataset_path,
        )
        gpu_type = CLUSTER_TO_GPU_TYPE[cluster]

        bs_config = BatchSizeConfig(
            sequence_length=seq_len,
            world_size=num_nodes * GPUS_PER_NODE,
            global_batch_size_tokens=global_batch_size,
            gpu_type=gpu_type,  # used to double microbatch size for B200s
        )
        if get_local_rank() == 0:
            print("Batch size config (before overrides):")
            print(bs_config)

        dp_shard_degree = GPUS_PER_NODE // (bs_config.cp_degree or 1)
        if not dp_shard_degree > 0:
            raise OLMoConfigurationError(f"dp_shard_degree ({dp_shard_degree}) must be positive.")

        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

        cp_config = (
            (
                TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
                if dataset_config.generate_doc_lengths  # only use llama3 if we're masking docs
                else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
            )
            if bs_config.cp_degree
            else None
        )

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=GPUS_PER_NODE  # try to keep communication w/in a node
            // (bs_config.cp_degree or 1),
        )

        # DIFF #3: opt-in performance / escape-hatch env knobs, ALL off by default so a
        # no-env run is built EXACTLY as AI2 (factory flash_2, factory dtype, bf16/no-FP8).
        # The Hopper launcher turns the perf ones on (FA3 + FP8) for the H200 runs;
        # context parallelism is automatic (BatchSizeConfig sets cp_degree at long seq).
        #  - OLMO_ATTN_BACKEND: e.g. flash_3 for FA3 packed-varlen on Hopper (vs AI2 flash_2).
        #  - OLMO_MODEL_DTYPE: override model storage dtype. Unset => factory.
        #  - OLMO_FP8=tensorwise|rowwise|rowwise_with_gw_hp: torchao float8 on the linears
        #    (H200 FP8 tensor cores). Unset => bf16 (AI2). tensorwise = fastest (cuBLAS +
        #    the tensorwise-only FSDP fp8 all-gather); rowwise = native CUTLASS kernel,
        #    more accurate but a bit slower & no fp8 all-gather. VALIDATE parity vs bf16.
        #    ALL attention + lm_head + embeddings stay high-precision (DeepSeek-V3 recipe).
        model_overrides: dict = {}
        if os.environ.get("OLMO_ATTN_BACKEND"):
            model_overrides["attn_backend"] = AttentionBackendName[os.environ["OLMO_ATTN_BACKEND"]]
        if os.environ.get("OLMO_MODEL_DTYPE"):
            model_overrides["dtype"] = (
                DType.bfloat16 if os.environ["OLMO_MODEL_DTYPE"] == "bfloat16" else DType.float32
            )
        model = TransformerConfig.olmo3_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            **model_overrides,
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
        )
        float8_config = None
        if os.environ.get("OLMO_FP8"):
            # FP8 on the feed-forward linears, but keep ALL attention projections in high
            # precision (DeepSeek-V3 recipe; see fp8_attention_ignores). lm_head + embeddings
            # are already safe (auto-excluded / not nn.Linear).
            float8_config = Float8Config(
                ao_recipe=AOFloat8LinearRecipe[os.environ["OLMO_FP8"]],
                modules_to_ignore=fp8_attention_ignores(model),
            )

        # DIFF #3 (cont.): optimizer by env. AI2's recipe uses SkipStepAdamW (skips a step when
        # the loss/grad-norm spikes past a rolling sigma band) for BOTH precisions. We keep that
        # for FP8 — the spike-prone path — so instability is caught instead of diverging, AND the
        # trainer auto-logs `optim/step skipped` (0/1 per step => averages to the skip frequency)
        # for any SkipStepOptimizer. For the stable BF16 baseline we opt into torch's FUSED AdamW
        # (single fused CUDA kernel) for speed. NB SkipStepAdamW == AdamW whenever it isn't
        # skipping, so the BF16-vs-FP8 update rule only diverges exactly when FP8 spikes (which the
        # skip metric flags). OLMO_OPTIM default = skip_step, so a no-env run stays AI2-bit-for-bit;
        # olmocore/run.sh wires bf16 -> fused_adamw and fp8 -> skip_step.  weight_decay=0.0 here is
        # the SFT recipe (different from pretraining).
        if os.environ.get("OLMO_OPTIM", "skip_step") == "fused_adamw":
            optim_config = AdamWConfig(
                lr=8e-05, weight_decay=0.0, betas=(0.9, 0.95), fused=True
            )
        else:
            optim_config = SkipStepAdamWConfig(
                lr=8e-05, weight_decay=0.0, betas=(0.9, 0.95), compile=False
            )

        config = SFTConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[
                    script,
                    cmd,
                    run_name,
                    checkpoint,
                    cluster,
                    f"--seq_len={seq_len}",
                    f"--num_nodes={num_nodes}",
                    f"--global_batch_size={global_batch_size}",
                    f"--budget={budget}",
                    f"--workspace={workspace}",
                    f"--dataset_path={dataset_path}",
                    *overrides,
                ],
                cluster=cluster,
                num_nodes=num_nodes,
                budget=budget,
                workspace=workspace,
            ),
            model=model,
            dataset=None,
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=bs_config.global_batch_size_tokens, seed=34521, num_workers=4
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=bs_config.rank_microbatch_size_tokens,
                max_sequence_length=bs_config.sequence_length,
                z_loss_multiplier=None,
                compile_model=True,
                float8_config=float8_config,  # DIFF #3: None unless OLMO_FP8 set (=> AI2 bf16)
                optim=optim_config,  # fused AdamW (bf16) | SkipStepAdamW (fp8) — see OLMO_OPTIM above
                dp_config=dp_config,
                cp_config=cp_config,
                ac_config=ac_config,
                scheduler=LinearWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.0,  # lr drops all the way to 0.0 at the end
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"{root_dir}/checkpoints/{user_name}/olmo-sft/{run_name}",
                load_strategy=LoadStrategy.never,  # we manually load the checkpoint below
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.epochs(3),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000, ephemeral_save_interval=500, save_async=True
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    # entity=None => the user's own default W&B entity (NOT ai2-llm, which only
                    # AI2 members can write to). Override with WANDB_ENTITY / WANDB_PROJECT.
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    project=os.environ.get("WANDB_PROJECT", "olmo3-7b-sft"),
                    # On when a key is provided (entrypoint.sh forces WANDB_MODE=offline otherwise).
                    # Enabling this is what surfaces `optim/step skipped` (the FP8 skip frequency)
                    # and the loss curves; upstream shipped it disabled for the Beaker path.
                    enabled=bool(os.environ.get("WANDB_API_KEY")),
                    cancel_check_interval=10,
                ),
            ),
            init_seed=init_seed,
        ).merge(overrides)

        config.dataset = dataset_config

        print(config)

        return config


def train(checkpoint: str, config: SFTConfig, no_save_tokenizer: bool):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    if config.dataset is not None:
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = config.trainer.build(train_module, data_loader)

        if not no_save_tokenizer and get_rank() == 0:
            tokenizer_path = join_path(get_parent(dataset.paths[0]), "tokenizer")
            if not dir_is_empty(tokenizer_path):
                log.info("Saving tokenizer...")
                destination_path = join_path(trainer.save_folder, "tokenizer")
                if not dir_is_empty(destination_path):
                    log.info(f"Tokenizer already exists: {destination_path}")
                else:
                    log.info(f"Saving tokenizer to {destination_path}")
                    copy_dir(tokenizer_path, destination_path)

        # Record the config to W&B/Comet and each checkpoint dir.
        config_dict = config.as_config_dict()
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

        # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
        log.info("Loading checkpoint...")
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            log.info(
                f"No checkpoint found in save folder '{trainer.save_folder}', attempting to load from pretraining checkpoint '{checkpoint}'"
            )
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        else:
            log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

        # Train.
        trainer.fit()
    else:
        log.error(f"Config dataset is None: {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT the 7B model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s dry_run test my-dataset-name /path/to/ckpt ai2/cluster
  python %(prog)s launch run01 OpenThoughts3-1.2M /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/jupiter-cirrascale-2 --seq_len=4096 --num_nodes=2 --launch.priority=high  --launch.follow=false
""",
    )

    # Subcommand
    parser.add_argument(
        "cmd",
        choices=["launch", "train", "dry_run"],
        help="Subcommand to run",
    )

    # Positional arguments
    parser.add_argument(
        "run_name",
        help="The name of the run. Used for the run name in W&B/Comet and the checkpoint dir.",
    )
    parser.add_argument("pretrain_checkpoint", help="Path to the pretraining checkpoint to load.")
    parser.add_argument(
        "cluster", help="The Beaker cluster to use (e.g., 'ai2/jupiter-cirrascale-2')."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="The maximum sequence length to use.",
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--num_nodes", type=int, help="The number of nodes to use.", default=DEFAULT_NUM_NODES
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Disable saving the dataset's tokenizer in the model directory.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="The global batch size in tokens.",
        default=64 * DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--budget", help="The beaker budget to use.")
    parser.add_argument("--workspace", help="The workspace to run in.")
    parser.add_argument("--dataset_path", help="The path to the pre-tokenized SFT dataset.")

    # Parse known args to get positional arguments and cmd
    args, overrides = parser.parse_known_args()

    # Prepare the environment for the given command.
    if args.cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif args.cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(args.cmd)

    # Build the config, applying any overrides.
    config = SFTConfig.build(
        script=sys.argv[0],
        cmd="train",
        run_name=args.run_name,
        checkpoint=args.pretrain_checkpoint,
        cluster=args.cluster,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        global_batch_size=args.global_batch_size,
        overrides=overrides,
        budget=args.budget,
        workspace=args.workspace,
        dataset_path=args.dataset_path,
    )

    # Print the config for debugging and then execute the command.
    if get_local_rank() == 0:
        print(config)

    if args.cmd == "dry_run":
        pass
    elif args.cmd == "launch":
        raise NotImplementedError(  # DIFF #7: no Beaker/Gantry launch in the local copy
            "'launch' is disabled in the local copy; run 'train' under torchrun."
        )
    elif args.cmd == "train":
        try:
            train(args.pretrain_checkpoint, config, args.no_save_tokenizer)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(args.cmd)
