from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_interp.config import RunConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_run_config_kwargs() -> dict:
    return {
        "run_name": "test_run",
        "model": {
            "id": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "dtype": "float32",
            "device": "cuda",
            "attn_implementation": "sdpa",
        },
        "capture": {
            "layers": [0, 1, 2],
            "top_k_logprobs": 20,
        },
        "generation": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 768,
            "seed": 0,
        },
        "storage_root": "runs/test_run",
    }


def test_loads_local_config():
    config = load_config(REPO_ROOT / "configs" / "qwen3_0p6b_local.yaml")
    assert config.run_name == "qwen3_0p6b_local"
    assert config.model.dtype == "float32"
    assert config.capture.layers == list(range(28))
    assert config.generation.temperature == 0.6
    assert config.capture.mlp.dense_layers == [7, 14, 21, 27]
    assert config.capture.mlp.topk_layers == []
    assert config.capture.mlp.top_k == 64
    assert config.capture.mlp.capture_validation_extras is False


def test_loads_cloud_config():
    config = load_config(REPO_ROOT / "configs" / "qwen3_8b_cloud.yaml")
    assert config.model.dtype == "bfloat16"
    assert config.capture.layers == list(range(36))
    assert config.capture.mlp.dense_layers == [9, 18, 27, 35]
    assert config.capture.mlp.topk_layers == list(range(36))


def test_valid_config_round_trips():
    config = RunConfig.model_validate(_base_run_config_kwargs())
    assert config.capture.layers == [0, 1, 2]


def test_rejects_float16():
    kwargs = _base_run_config_kwargs()
    kwargs["model"]["dtype"] = "float16"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_rejects_flash_attention():
    kwargs = _base_run_config_kwargs()
    kwargs["model"]["attn_implementation"] = "flash_attention_2"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_rejects_bad_device():
    kwargs = _base_run_config_kwargs()
    kwargs["model"]["device"] = "tpu"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_rejects_greedy_temperature():
    kwargs = _base_run_config_kwargs()
    kwargs["generation"]["temperature"] = 0.0
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_rejects_duplicate_layers():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["layers"] = [0, 1, 1]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_rejects_negative_layer():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["layers"] = [-1, 0]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_layers_are_sorted():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["layers"] = [2, 0, 1]
    config = RunConfig.model_validate(kwargs)
    assert config.capture.layers == [0, 1, 2]


def test_config_is_frozen():
    config = RunConfig.model_validate(_base_run_config_kwargs())
    with pytest.raises(ValidationError):
        config.run_name = "mutated"


def test_mlp_capture_defaults_to_none():
    config = RunConfig.model_validate(_base_run_config_kwargs())
    assert config.capture.mlp is None


def test_mlp_capture_config_defaults():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [0]}
    config = RunConfig.model_validate(kwargs)
    assert config.capture.mlp.dense_layers == [0]
    assert config.capture.mlp.topk_layers == []
    assert config.capture.mlp.top_k == 64
    assert config.capture.mlp.capture_validation_extras is False


def test_mlp_capture_allows_overlapping_dense_and_topk_layers():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [0, 1], "topk_layers": [0, 1, 2]}
    config = RunConfig.model_validate(kwargs)
    assert config.capture.mlp.dense_layers == [0, 1]
    assert config.capture.mlp.topk_layers == [0, 1, 2]


def test_mlp_capture_rejects_duplicate_dense_layers():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [0, 0]}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_mlp_capture_rejects_negative_topk_layer():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"topk_layers": [-1]}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_mlp_capture_rejects_nonpositive_top_k():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [0], "top_k": 0}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(kwargs)


def test_mlp_capture_layers_are_sorted():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [2, 0, 1]}
    config = RunConfig.model_validate(kwargs)
    assert config.capture.mlp.dense_layers == [0, 1, 2]


def test_mlp_capture_config_is_frozen():
    kwargs = _base_run_config_kwargs()
    kwargs["capture"]["mlp"] = {"dense_layers": [0]}
    config = RunConfig.model_validate(kwargs)
    with pytest.raises(ValidationError):
        config.capture.mlp.top_k = 1
