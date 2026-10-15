from pathlib import Path

import pytest

from llm_interp.config import load_config
from llm_interp.estimate import estimate_bytes

pytestmark = pytest.mark.gpu

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_estimate_bytes_matches_hand_computed_total(qwen3_run_config):
    config = qwen3_run_config.model_copy(
        update={"capture": qwen3_run_config.capture.model_copy(update={"layers": [0, 1], "mlp": None})}
    )
    result = estimate_bytes(config, num_tokens=1000)

    hidden_size = 1024  # Qwen3-0.6B
    dtype_bytes = 4  # float32
    expected_per_token = (
        2 * hidden_size * dtype_bytes  # resid_pre for 2 layers
        + hidden_size * dtype_bytes  # final_residual
        + 8  # positions
        + 8  # token_ids
        + config.capture.top_k_logprobs * 4  # top_logprob_values
        + config.capture.top_k_logprobs * 8  # top_logprob_indices
        + 4  # realized_logprob
    )
    assert result.bytes_per_token == expected_per_token
    assert result.num_tokens == 1000
    assert result.total_bytes == expected_per_token * 1000


def test_estimate_bytes_scales_with_num_layers(qwen3_run_config):
    base_capture = qwen3_run_config.capture.model_copy(update={"mlp": None})
    one_layer = qwen3_run_config.model_copy(update={"capture": base_capture.model_copy(update={"layers": [0]})})
    two_layers = qwen3_run_config.model_copy(update={"capture": base_capture.model_copy(update={"layers": [0, 1]})})

    r1 = estimate_bytes(one_layer, num_tokens=100)
    r2 = estimate_bytes(two_layers, num_tokens=100)
    assert r2.bytes_per_token - r1.bytes_per_token == 1024 * 4  # one more fp32 resid_pre layer


def test_estimate_bytes_mlp_dense_matches_addendum_reference_figures(qwen3_run_config):
    """Addendum A reference figures for Qwen3-0.6B, fp32 (stated in decimal KB,
    i.e. /1000 — this asserts the exact byte counts those round to)."""
    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [7], "topk_layers": [], "capture_validation_extras": False}
                    ),
                }
            )
        }
    )
    result = estimate_bytes(config, num_tokens=1)
    assert result.per_site_bytes["mlp_act/7"] == 12288  # 3072 * 4 bytes; addendum: ~12.3 KB (decimal)
    assert result.per_site_bytes["mlp_gate_sign/7"] == 384  # ceil(3072/8); addendum: 384 B


def test_estimate_bytes_mlp_topk_matches_addendum_reference_figures(qwen3_run_config):
    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [], "topk_layers": [7], "top_k": 64, "capture_validation_extras": False}
                    ),
                }
            )
        }
    )
    result = estimate_bytes(config, num_tokens=1)
    topk_bytes = (
        result.per_site_bytes["mlp_topk_index/7"]
        + result.per_site_bytes["mlp_topk_value/7"]
        + result.per_site_bytes["mlp_topk_regime/7"]
    )
    assert topk_bytes == 448  # addendum: ~448 B per token-layer, k=64, fp32


def test_estimate_bytes_mlp_dense_8b_bf16_matches_addendum_reference_figures():
    """8B bf16 cloud config: dense act ~24.6 KB/token-layer (decimal), ~885 GB
    per 1M tokens over 36 layers; topk ~320 B/token-layer, ~11.5 GB per 1M
    tokens over 36 layers. AutoConfig only — the 8B weights are never fetched."""
    config = load_config(REPO_CONFIGS / "qwen3_8b_cloud.yaml")
    result = estimate_bytes(config, num_tokens=1_000_000)

    dense_layer = config.capture.mlp.dense_layers[0]
    dense_bytes_per_token_layer = result.per_site_bytes[f"mlp_act/{dense_layer}"]
    assert dense_bytes_per_token_layer == 24576  # 12288 * 2 bytes (bf16)

    topk_layer = config.capture.mlp.topk_layers[0]
    topk_bytes_per_token_layer = (
        result.per_site_bytes[f"mlp_topk_index/{topk_layer}"]
        + result.per_site_bytes[f"mlp_topk_value/{topk_layer}"]
        + result.per_site_bytes[f"mlp_topk_regime/{topk_layer}"]
    )
    assert topk_bytes_per_token_layer == 320

    # The addendum's "~885 GB / 1M tokens" and "~11.5 GB / 1M tokens" figures are
    # the per-layer byte cost scaled to all 36 layers (a comparison point for
    # choosing topk over all-dense at cloud scale) — not this config's actual
    # dense_layers count (4). Scale explicitly to 36 to match that comparison.
    assert dense_bytes_per_token_layer * 36 * 1_000_000 == 884_736_000_000  # ~885 GB (decimal)
    assert topk_bytes_per_token_layer * 36 * 1_000_000 == 11_520_000_000  # ~11.5 GB (decimal)
