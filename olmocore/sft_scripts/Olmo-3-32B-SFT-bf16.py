"""
32B SFT — bf16-only, FP8-REMOVED variant of `Olmo-3-32B-SFT-local.py`.

Identical to `Olmo-3-32B-SFT-local.py` EXCEPT all FP8 (torchao float8) code paths are removed —
imports, `fp8_attention_ignores`, the `float8_config` block, and the fp8->skip_step optimizer
coupling. Rationale: the Olmo-3 authors flagged FP8 SFT as too reckless, so this script has NO FP8
surface at all (not merely defaulted off), and it RAISES if `OLMO_FP8` / `OLMO_FP8_FSDP_ALLGATHER`
is set so a stray env var can't silently re-enable it. Model training is bf16 throughout (the
model/optimizer dtype behavior is unchanged from the base script's no-FP8 path).

Select it via `SFT_SCRIPT_NAME=Olmo-3-32B-SFT-bf16.py` (run.sh resolves it under sft_scripts/).

Everything else is byte-for-byte the base 32B script — Beaker stubs, hsdp, selective AC,
compile_model=True, generate_doc_lengths=True, YaRN(factor=8, 8192), and the env-gated knobs below —
so re-diff against `Olmo-3-32B-SFT-local.py` when bumping the pin.

Retained env knobs (all opt-in / off by default unless noted):
  1. Beaker decoupling via local stubs (get_root_dir -> $OLMO_SFT_SAVE_ROOT; `launch` disabled).
  2. GPUS_PER_NODE from torchrun's LOCAL_WORLD_SIZE.
  3. OLMO_ATTN_BACKEND=flash_2|flash_3 ; OLMO_MODEL_DTYPE ; OLMO_USE_SINK(=1)/OLMO_SINK_INIT
     (per-head attention sink; requires flash_2/flash_3).
  4. OLMO_FUSED_LCE=1 : Liger fused-linear CE (off by default; ~6.5% higher loss measured here).
  5. OLMO_CP_STYLE (default ring): ring (llama3, doc-mask-aware) or ulysses (all-to-all).
  6. OLMO_AC_BUDGET : activation-checkpointing mode (unset = selected_modules; <0..1> = budget; none).
  7. OLMO_SAVE_INTERVAL / OLMO_EPHEMERAL_INTERVAL / OLMO_KEEP_LAST_CKPTS : checkpoint cadence + retention.
  8. OLMO_OPTIM (default fused_adamw here) : fused AdamW or skip_step (SkipStepAdamW spike protection,
     independent of FP8); OLMO_OPTIM_DTYPE=bf16 stores Adam moments in bf16 (skip_step path only).
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
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_fs_local_rank, get_local_rank, get_rank
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
from olmo_core.io import clear_directory, copy_dir, dir_is_empty, get_parent, join_path, list_directory
from olmo_core.nn.attention import AttentionBackendName  # DIFF #4 (env attn backend)
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, SkipStepAdamWConfig
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
from olmo_core.train.callbacks.callback import Callback
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
# Per-rank token budget that drives cp_degree (cp = smallest pow2 with seq_len/cp <= this). AI2's value:
# 16384 = "max this config handles on an H100" (80 GB); the code below doubles it for B200. It's an
# empirical tuning heuristic, NOT a hard limit — H200 (141 GB) comfortably handles 32768. Override with
# OLMO_MAX_TOKENS_PER_RANK to halve CP (e.g. cp 4->2 at seq 65536): 2x tokens/GPU + 2x data-parallel.
MAX_RANK_MICROBATCH_SIZE_TOKENS = int(os.environ.get("OLMO_MAX_TOKENS_PER_RANK", "16384"))


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
        # NOTE (local fork change): world_size is NOT required to be a power of two. AI2 upstream asserted it
        # (their clusters are power-of-two), but the real constraints are just the divisibility checks below.
        # Dropping it lets a non-pow2 node count work (e.g. 3 nodes -> world_size 24), so ONE global batch can
        # serve 2/3/4 nodes (1.5M tok -> grad_accum 6/4/3). The 7B SFT script is a separate, frozen file and
        # is unaffected.

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
        assert self.global_batch_size_tokens % dp_world_size == 0, (
            f"global_batch_size_tokens ({self.global_batch_size_tokens}) must be divisible by the data-parallel "
            f"world size ({dp_world_size} = world_size {self.world_size} / cp {cp_factor}); pick a global batch "
            f"that is a multiple of {dp_world_size}."
        )
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        # rank microbatch = the whole sequences that fit the per-rank cap (max_tokens_per_rank x cp). With
        # intra-doc masking this stays at 1x seq_len for our SFT (cap == seq_len); the general form also
        # supports B200's 2x cap. grad_accum is then computed DIRECTLY (rank_batch / microbatch), NOT by the
        # old power-of-two doubling loop — so non-pow2 node counts work (3 nodes/dp_world=6 -> grad_accum 4,
        # 2 nodes -> 6, 4 nodes -> 3 at a 1.5M global batch). It is IDENTICAL to the old loop whenever the
        # ratio is already a power of two (e.g. all the 7B/pow2 shapes), so those configs are unchanged.
        microbatch_cap = max_tokens_per_rank * cp_factor
        seqs_per_microbatch = max(1, microbatch_cap // self.sequence_length)
        self.rank_microbatch_size_tokens = seqs_per_microbatch * self.sequence_length
        assert rank_batch_size_tokens % self.rank_microbatch_size_tokens == 0, (
            f"rank batch ({rank_batch_size_tokens}) must be a whole multiple of the rank microbatch "
            f"({self.rank_microbatch_size_tokens}); with {dp_world_size} DP ranks, pick a global batch that "
            f"is a multiple of {dp_world_size * self.rank_microbatch_size_tokens}."
        )
        self.grad_accum_steps = rank_batch_size_tokens // self.rank_microbatch_size_tokens
        if self.grad_accum_steps > 1:
            log.info(
                f"rank batch {rank_batch_size_tokens} tok / microbatch {self.rank_microbatch_size_tokens} "
                f"-> grad_accum_steps={self.grad_accum_steps} (dp_world={dp_world_size}, cp={cp_factor})"
            )

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


def _read_hf_config(identifier: str) -> dict:
    """Read a HF model's config.json (mirrors TokenizerConfig.from_hf's fetch; used for rope_scaling)."""
    import json

    from cached_path import cached_path

    with cached_path(f"hf://{identifier}/config.json").open() as f:
        return json.load(f)


def _yarn_rope_scaling(hf_cfg: Optional[dict]) -> YaRNRoPEScalingConfig:
    """YaRN scaling taken from the model's OWN config.json ``rope_scaling`` (so the factor matches how
    the model was trained — e.g. factor 32 -> 262144 for the deepseek-transplant Olmo3), falling back
    to the AI2 default (factor 8 -> 65536) when no HF model is given. olmo-core derives YaRN's
    ``attention_factor`` from ``factor``, matching HF's stored value."""
    if hf_cfg:
        rs = hf_cfg.get("rope_scaling") or {}
        if str(rs.get("rope_type", "")).lower() == "yarn":
            return YaRNRoPEScalingConfig(
                factor=float(rs["factor"]),
                beta_fast=float(rs.get("beta_fast", 32)),
                beta_slow=float(rs.get("beta_slow", 1)),
                old_context_len=int(rs.get("original_max_position_embeddings", 8192)),
            )
    return YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)


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


@dataclass
class KeepLastNCheckpoints(Callback):
    """DIFF #9: cap PERSISTENT (save_interval) checkpoints on disk to `keep_last`, deleting the oldest
    as new ones are written. olmo-core's CheckpointerCallback only auto-prunes EPHEMERAL checkpoints,
    so persistent ones (distcp ~100 GB/7B, ~450 GB/32B) accumulate unbounded and blow a fixed storage
    budget. We delete only checkpoints older than the keep window (long finalized) — never the most
    recent N — so async saves and resume are unaffected. The final end-of-training checkpoint (saved
    at the last step, not an interval) counts toward the cap. keep_last<=0 disables. save_interval MUST
    match the CheckpointerCallback's so persistent saves are tagged (ephemeral steps are skipped)."""

    keep_last: int = 0
    save_interval: int = 1000
    _persistent: List[str] = field(default_factory=list)

    def _persistent_step(self, path) -> Optional[int]:
        """The step if `path` is a PERSISTENT checkpoint, else None — derived from the checkpoint
        DIRECTORY NAME (step<N>), NOT self.step. Checkpoint saves are async by default
        (save_async=backend_supports_cpu()), so post_checkpoint_saved fires from the save future's
        done-callback LONG after self.step (==trainer.global_step) has advanced past the save_interval
        boundary. Keying retention off self.step would (almost) never match a persistent step, so
        persistent checkpoints would never be tracked or pruned and would accumulate unbounded."""
        name = str(path).rstrip("/").split("/")[-1]
        if not (name.startswith("step") and name[4:].isdigit()):
            return None
        step = int(name[4:])
        if self.save_interval > 0 and step % self.save_interval == 0:
            return step
        max_steps = getattr(self.trainer, "max_steps", None)  # the end-of-training checkpoint
        return step if (max_steps is not None and step == max_steps) else None

    def pre_train(self):
        if self.keep_last <= 0:
            return
        try:  # on resume, seed from persistent checkpoints already on disk so they're pruned too
            found = []
            for p in list_directory(self.trainer.save_folder):
                name = str(p).rstrip("/").split("/")[-1]
                if name.startswith("step") and name[4:].isdigit() and int(name[4:]) % self.save_interval == 0:
                    found.append((int(name[4:]), str(p)))
            self._persistent = [p for _, p in sorted(found)]
            if self._persistent:
                log.info(f"[retention] tracking {len(self._persistent)} existing persistent checkpoint(s)")
        except Exception as e:  # noqa: BLE001
            log.warning(f"[retention] could not scan existing checkpoints: {e}")

    def post_checkpoint_saved(self, path):
        if self.keep_last <= 0 or self._persistent_step(path) is None:
            return  # ephemeral / unrecognized -> olmo-core's CheckpointerCallback manages those
        self._persistent.append(str(path))
        while len(self._persistent) > self.keep_last:
            old = self._persistent.pop(0)
            if get_fs_local_rank() == 0:
                log.info(f"[retention] pruning old persistent checkpoint {old} (keep_last={self.keep_last})")
                self.trainer.run_bookkeeping_op(
                    clear_directory, old, op_name=f"prune_checkpoint {old}", distributed=False
                )


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

        # Tokenizer: default dolma2, but OLMO_TOKENIZER_HF=<hf-model-id> reads vocab_size + eos/pad/bos
        # straight from that model's config.json (TokenizerConfig.from_hf). Required for non-dolma2
        # models — e.g. this stack's deepseek-transplant Olmo3 (vocab 129280, eos 1 / bos 0 / pad 2).
        # It sizes the embedding/lm_head (padded_vocab_size) and gives the eos id used for intra-doc
        # masking, so it MUST match both the base checkpoint and how the .npy data was tokenized.
        _hf_tok = os.environ.get("OLMO_TOKENIZER_HF")
        # Read the model's config.json once (when a HF model is given): its vocab drives the tokenizer
        # and its rope_scaling drives YaRN below, so both always match how the model was trained.
        _hf_cfg = _read_hf_config(_hf_tok) if _hf_tok else None
        tokenizer_config = TokenizerConfig.from_hf(_hf_tok) if _hf_tok else TokenizerConfig.dolma2()
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

        # DIFF #7: activation checkpointing. Reference = selected_modules (recompute every FFN, a
        # blunt rule). OLMO_AC_BUDGET=<0..1> switches to BUDGET mode: torch.compile's partitioner
        # solves the optimal save-vs-recompute split for that memory fraction (1=save all/recompute
        # nothing, 0=recompute all). AI2's long-context scripts use budget (7B=0.7, 32B=0.3, hybrid
        # SFT=0.1); the 96 GB/141 GB boxes are not memory-bound, so a HIGH budget (~0.8) recomputes
        # LESS = faster. OLMO_AC_BUDGET=none disables AC entirely (AI2's 7B long-context does this).
        # Budget mode relies on compile_model=True (we have it).
        _ac_budget = os.environ.get("OLMO_AC_BUDGET")
        if _ac_budget is None:
            ac_config = TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_modules,
                modules=["blocks.*.feed_forward"],
            )
        elif _ac_budget.lower() in ("none", "off"):
            ac_config = None
        else:
            ac_config = TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.budget,
                activation_memory_budget=float(_ac_budget),
            )

        # DIFF #6: context-parallel comm style. DEFAULT = ring (llama3, doc-mask-aware) — the PROVEN
        # path (it's what our earlier run trained on). OLMO_CP_STYLE=ulysses opts into all-to-all
        # (bandwidth-bound, PCIe-friendlier, AI2's long-context-SFT choice) — BUT on our config
        # (cp=4 + intra-doc masking + torch.compile) it tripped a device-side index assert in the
        # attention position/bucketize kernel (Ulysses passes FULL-sequence cu_doc_lens while the tensor
        # is seq-sharded to 16384/rank -> OOB index). So ulysses stays opt-in until that's debugged.
        # Ulysses needs n_heads % cp == 0 and cp within one node (cp<=GPUS_PER_NODE) for the all-to-all.
        _cp_style = os.environ.get("OLMO_CP_STYLE", "ring").lower()
        if not bs_config.cp_degree:
            cp_config = None
        elif _cp_style == "ulysses":
            if bs_config.cp_degree > GPUS_PER_NODE:
                raise OLMoConfigurationError(
                    f"ulysses cp_degree={bs_config.cp_degree} > GPUS_PER_NODE={GPUS_PER_NODE}: the "
                    "all-to-all would span nodes. Use more GPUs/node, lower SEQ_LEN, or raise the cap."
                )
            cp_config = TransformerContextParallelConfig.ulysses(degree=bs_config.cp_degree)
        elif _cp_style == "ring":
            cp_config = (
                TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
                if dataset_config.generate_doc_lengths  # llama3 is doc-mask-aware; zigzag isn't
                else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
            )
        else:
            raise OLMoConfigurationError(f"OLMO_CP_STYLE='{_cp_style}' (want ring|ulysses)")

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=GPUS_PER_NODE  # try to keep communication w/in a node
            // (bs_config.cp_degree or 1),
        )

        # Opt-in model-build env knobs, ALL off by default so a no-env run is built EXACTLY as AI2
        # (factory flash_2, factory dtype, bf16). Context parallelism is automatic (BatchSizeConfig
        # sets cp_degree at long seq). FP8 is NOT available here (see the guard above).
        #  - OLMO_ATTN_BACKEND: e.g. flash_3 for FA3 packed-varlen on Hopper (vs AI2 flash_2).
        #  - OLMO_MODEL_DTYPE: override model storage dtype. Unset => factory.
        #  - OLMO_USE_SINK=1 / OLMO_SINK_INIT: per-head attention sink (requires flash_2/flash_3).
        model_overrides: dict = {}
        if os.environ.get("OLMO_ATTN_BACKEND"):
            model_overrides["attn_backend"] = AttentionBackendName[os.environ["OLMO_ATTN_BACKEND"]]
        if os.environ.get("OLMO_MODEL_DTYPE"):
            model_overrides["dtype"] = (
                DType.bfloat16 if os.environ["OLMO_MODEL_DTYPE"] == "bfloat16" else DType.float32
            )
        # OLMO_USE_SINK=1 adds a per-head learnable attention sink to every layer (ported from the
        # olmo3_sink fork's fa3_sink.py: an exact re-normalization of flash's (out, softmax_lse)).
        # Works on the flash_2 AND flash_3 backends; run.sh defaults to flash_3 on Hopper (H100/H200)
        # and flash_2 elsewhere / for ring CP. Composes with Ulysses CP (OLMO_CP_STYLE=ulysses) but NOT
        # ring. OLMO_SINK_INIT sets the initial per-head logit: 0.0 = from-scratch; a strongly negative
        # value (e.g. -10.0) warm-starts from a checkpoint trained WITHOUT sinks (step 0 ~ no-op). When
        # warm-starting from a sink-baked HF checkpoint, the measured sinks round-trip through the
        # distcp converter automatically, so only use_sink is needed (leave OLMO_SINK_INIT unset).
        if os.environ.get("OLMO_USE_SINK") == "1":
            model_overrides["use_sink"] = True
            model_overrides["sink_init"] = float(os.environ.get("OLMO_SINK_INIT", "0.0"))
        # YaRN scaling from the model's own config.json (factor 32 -> 262144 for this model), or the
        # AI2 default (factor 8 -> 65536) when no HF model is given. Reading it from the config means
        # SEQ_LEN can be anything up to the model's trained max without a hardcoded factor mismatch.
        model = TransformerConfig.olmo3_32B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            **model_overrides,
        ).with_rope_scaling(_yarn_rope_scaling(_hf_cfg))
        # DIFF #5: OLMO_FUSED_LCE=1 opts into Liger fused-linear cross-entropy (no materialized
        # (T, vocab) logits, ~10 GB saved at seq 65536). DEFAULT OFF — measured on our setup it gives
        # ~6.5% HIGHER loss than the materialized reference (a Liger reduction-normalization
        # difference, NOT logit precision) AND lower MFU/throughput (per-step device sync in Liger's
        # backward breaks overlap; no memory-bandwidth win on a comm-bound, non-lm-head-bound step).
        # Materialized matches AI2's Olmo-3-7B-SFT.py reference + BF16. Use fused ONLY when genuinely
        # memory-constrained (e.g. 32B) and validate convergence first. liger-kernel is in the image.
        if os.environ.get("OLMO_FUSED_LCE") == "1":
            from olmo_core.nn.lm_head import LMLossImplementation
            model.lm_head.loss_implementation = LMLossImplementation.fused_linear
        # FP8 is intentionally REMOVED from this script (the Olmo-3 authors flagged FP8 SFT as too
        # reckless). Training is bf16 throughout: the train module gets no float8_config. Fail LOUDLY
        # if a stray env var tries to re-enable FP8, rather than silently ignoring it.
        if os.environ.get("OLMO_FP8") or os.environ.get("OLMO_FP8_FSDP_ALLGATHER"):
            raise OLMoConfigurationError(
                "OLMO_FP8 / OLMO_FP8_FSDP_ALLGATHER are set, but this is the FP8-free bf16 script "
                "(Olmo-3-32B-SFT-bf16.py) which has no FP8 code path. Unset them, or use "
                "Olmo-3-32B-SFT-local.py if you really intend FP8."
            )

        # Optimizer by env. This bf16 script defaults to torch's FUSED AdamW (single fused CUDA kernel,
        # fp32 state) — the stable bf16 baseline. OLMO_OPTIM=skip_step opts into SkipStepAdamW, which
        # skips a step when the loss/grad-norm spikes past a rolling sigma band and auto-logs
        # `optim/step skipped` (0/1 per step); it equals AdamW whenever it isn't skipping. (This is
        # spike protection, independent of FP8 — which this script does not support.) weight_decay=0.0
        # is the SFT recipe (different from pretraining).
        # OLMO_OPTIM_DTYPE=bf16 stores the Adam MOMENTS (exp_avg/exp_avg_sq) in bf16 instead of fp32 —
        # halves optimizer VRAM + checkpoint while KEEPING fp32 master weights. Only the skip_step
        # optimizer exposes a state dtype. The bf16 2nd moment loses precision in the Adam denominator
        # — VALIDATE loss parity vs fp32. Default off.
        _opt_dt = DType.bfloat16 if os.environ.get("OLMO_OPTIM_DTYPE") in ("bf16", "bfloat16") else None
        if os.environ.get("OLMO_OPTIM", "fused_adamw") == "fused_adamw":
            if _opt_dt is not None:
                log.warning("OLMO_OPTIM_DTYPE applies only to the skip_step optimizer; fused AdamW keeps fp32 state")
            optim_config = AdamWConfig(
                lr=8e-05, weight_decay=0.0, betas=(0.9, 0.95), fused=True
            )
        else:
            optim_config = SkipStepAdamWConfig(
                lr=8e-05, weight_decay=0.0, betas=(0.9, 0.95), compile=False, dtype=_opt_dt
            )

        # Checkpoint cadence + retention (env-tunable). Persistent every OLMO_SAVE_INTERVAL steps,
        # ephemeral (rotating resume points) every OLMO_EPHEMERAL_INTERVAL. OLMO_KEEP_LAST_CKPTS caps
        # PERSISTENT checkpoints on disk (olmo-core only auto-prunes ephemeral); run.sh defaults it per
        # model size since distcp checkpoints are ~100 GB (7B) / ~450 GB (32B) and the budget is ~1 TB.
        # 0 = keep all. ephemeral_interval must be < save_interval (olmo-core asserts this).
        _save_interval = int(os.environ.get("OLMO_SAVE_INTERVAL", "1000"))
        _ephemeral_interval = int(os.environ.get("OLMO_EPHEMERAL_INTERVAL", "500"))
        _keep_last = int(os.environ.get("OLMO_KEEP_LAST_CKPTS", "0"))

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
                optim=optim_config,  # fused AdamW (default) | SkipStepAdamW — see OLMO_OPTIM above
                dp_config=dp_config,
                cp_config=cp_config,
                ac_config=ac_config,
                scheduler=CosWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.1,  # floor the LR at 0.1*peak (5e-5 -> 5e-6 final), NOT 0. Deliberate, for
                                  # two reasons: (1) the data is rich enough that we expect to keep
                                  # improving to the very end, so we don't want to kill the LR; (2) we may
                                  # model-merge (TIES / checkpoint soup), and a non-collapsed late
                                  # trajectory keeps the endpoint in a broad, mergeable basin. Cosine (vs
                                  # linear) still gives a gentler tail: ~0.19*peak at 80% vs linear's 0.28.
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
                    save_interval=_save_interval,
                    ephemeral_save_interval=_ephemeral_interval,
                    save_async=True,
                ),
            )
            .with_callback(
                "checkpoint_retention",
                KeepLastNCheckpoints(keep_last=_keep_last, save_interval=_save_interval),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    # entity=None => the user's own default W&B entity (NOT ai2-llm, which only
                    # AI2 members can write to). Override with WANDB_ENTITY / WANDB_PROJECT.
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    project=os.environ.get("WANDB_PROJECT", "olmo3-32b-sft"),
                    # On when a key is provided (entrypoint.sh forces WANDB_MODE=offline otherwise).
                    # Enabling this is what surfaces `optim/step skipped` (the skip_step skip frequency)
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
        description="SFT the 32B model.",
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
