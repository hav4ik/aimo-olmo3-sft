"""Recipe-side unit tests for the RMSNorm -> FusedRMSNorm swap in Olmo-3-32B-SFT-bf16.py.

The swap dodges the inductor persistent-reduction shared-memory OOM on GPUs whose opt-in smem/block
can't hold the compiled ~196 KB RMSNorm reduction (RTX 6000 ~99 KB, A100 ~163 KB). These tests pin:

  * the tri-state env-flag DECISION logic (auto / force-on / force-off, tolerant spellings);
  * that ALL wide norms are covered — block, lm_head AND q_norm. In olmo3_32B ``use_head_qk_norm=False``
    so q_norm is built over the full n_heads*head_dim=5120 projection (as wide as block.layer_norm), NOT
    per-head; missing it would re-introduce the OOM. All three sites share ONE LayerNormConfig object, so
    the swap must ``replace`` (fresh object) rather than mutate in place;
  * that the swapped config actually builds a FusedRMSNorm (a green config-string test must not hide a
    non-constructible norm);
  * that there is no other wide norm site (``embedding_norm`` is None) silently left on stock ``rms``.

Config-only (no GPU / no compile), so it runs anywhere ``olmo_core`` imports.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("olmo_core")

from olmo_core.nn.layer_norm import FusedRMSNorm, LayerNormType  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

_SFT_SCRIPT = Path(__file__).resolve().parents[1] / "sft_scripts" / "Olmo-3-32B-SFT-bf16.py"


def _load_sft():
    spec = importlib.util.spec_from_file_location("olmo3_32b_sft_bf16", _SFT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sft = _load_sft()


def _cfg() -> TransformerConfig:
    # config only — no weights are allocated, so this is cheap and needs no GPU.
    return TransformerConfig.olmo3_32B(vocab_size=100352)


def _names(cfg):
    return (
        cfg.block.layer_norm.name,
        cfg.lm_head.layer_norm.name,
        cfg.block.sequence_mixer.qk_norm.name,
    )


# ---- _env_bool: the tri-state flag parser -----------------------------------------------------


@pytest.mark.parametrize(
    "raw, expect",
    [
        (None, None),
        ("", None),  # empty -> auto (matches train.py dropping empty values)
        ("garbage", None),  # unrecognized -> auto (safe direction)
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("  on ", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("no", False),
    ],
)
def test_env_bool(raw, expect):
    assert sft._env_bool(raw) is expect


# ---- the swap decision + targets --------------------------------------------------------------


@pytest.mark.parametrize(
    "small_smem, env_val, expect_swap",
    [
        (True, None, True),  # auto-on: RTX 6000 / A100 (small smem, no override)
        (False, None, False),  # auto-off: Hopper / B200 (big smem, no override)
        (False, "1", True),  # force on regardless of GPU
        (True, "0", False),  # force off regardless of GPU
        (False, "true", True),  # tolerant spelling, force on
        (True, "off", False),  # tolerant spelling, force off
        (True, "", True),  # empty -> auto -> on (small smem)
        (True, "garbage", True),  # unrecognized -> auto -> on (small smem)
        (False, "garbage", False),  # unrecognized -> auto -> off (big smem)
    ],
)
def test_swap_decision_and_targets(small_smem, env_val, expect_swap):
    cfg = _cfg()
    applied = sft._maybe_use_fused_rmsnorm(cfg, small_smem, env_val)
    assert applied is expect_swap

    want = LayerNormType.fused_rms if expect_swap else LayerNormType.rms
    # ALL wide norms flip together (block, lm_head, AND q/k norm — q_norm is 5120-wide here).
    assert _names(cfg) == (want, want, want)


def test_all_three_norm_sites_share_one_config_and_all_get_swapped():
    """Precondition: block.layer_norm, lm_head.layer_norm and qk_norm are the SAME object.

    The swap must replace (not mutate) so it produces a fresh shared config with all three flipped —
    covering the wide q_norm — while never depending on in-place mutation of the original.
    """
    cfg = _cfg()
    assert cfg.block.layer_norm is cfg.lm_head.layer_norm is cfg.block.sequence_mixer.qk_norm

    applied = sft._maybe_use_fused_rmsnorm(cfg, small_smem=True, env_val=None)
    assert applied is True
    assert _names(cfg) == (LayerNormType.fused_rms,) * 3
    # the original shared object was replaced, not mutated in place.
    assert cfg.block.layer_norm.name == LayerNormType.fused_rms


def test_swap_preserves_other_norm_fields():
    """Only `name` changes — eps/bias/full_precision must survive the replace()."""
    cfg = _cfg()
    before = cfg.block.layer_norm
    orig = (before.eps, before.bias, before.full_precision, before.elementwise_affine)

    sft._maybe_use_fused_rmsnorm(cfg, small_smem=True, env_val=None)

    after = cfg.block.layer_norm
    assert (after.eps, after.bias, after.full_precision, after.elementwise_affine) == orig
    assert after.name == LayerNormType.fused_rms


def test_no_swap_leaves_config_untouched():
    cfg = _cfg()
    block_before = cfg.block.layer_norm
    applied = sft._maybe_use_fused_rmsnorm(cfg, small_smem=False, env_val=None)
    assert applied is False
    assert cfg.block.layer_norm is block_before  # same object, not replaced
    assert _names(cfg) == (LayerNormType.rms,) * 3


def test_no_other_wide_norm_site_left_behind():
    """embedding_norm is a real d_model-wide norm slot; it must be None for this config (else it would
    ALSO OOM at compile and the swap would be incomplete)."""
    cfg = _cfg()
    assert cfg.embedding_norm is None


def test_swapped_config_actually_builds_a_fused_rmsnorm():
    """A green config-string test must not hide a non-constructible norm — build the swapped configs."""
    pytest.importorskip("flash_attn")
    cfg = _cfg()
    sft._maybe_use_fused_rmsnorm(cfg, small_smem=True, env_val=None)
    # build at the two real widths: block/lm_head over d_model, and q_norm over n_heads*head_dim.
    block_norm = cfg.block.layer_norm.build(size=cfg.d_model, init_device="cpu")
    qk_norm = cfg.block.sequence_mixer.qk_norm.build(size=cfg.d_model, init_device="cpu")
    assert isinstance(block_norm, FusedRMSNorm)
    assert isinstance(qk_norm, FusedRMSNorm)
