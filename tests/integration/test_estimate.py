import pytest

from llm_interp.estimate import estimate_bytes

pytestmark = pytest.mark.gpu


def test_estimate_bytes_matches_hand_computed_total(qwen3_run_config):
    config = qwen3_run_config.model_copy(update={"capture": qwen3_run_config.capture.model_copy(update={"layers": [0, 1]})})
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
    one_layer = qwen3_run_config.model_copy(update={"capture": qwen3_run_config.capture.model_copy(update={"layers": [0]})})
    two_layers = qwen3_run_config.model_copy(update={"capture": qwen3_run_config.capture.model_copy(update={"layers": [0, 1]})})

    r1 = estimate_bytes(one_layer, num_tokens=100)
    r2 = estimate_bytes(two_layers, num_tokens=100)
    assert r2.bytes_per_token - r1.bytes_per_token == 1024 * 4  # one more fp32 resid_pre layer
