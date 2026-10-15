import os

# Must be set before the first CUDA context init for deterministic cuBLAS algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import pytest
import torch

from llm_interp.config import RunConfig, load_config
from llm_interp.model_loading import build_adapter, load_model_and_tokenizer
from llm_interp.validate import ValidationContext

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def qwen3_run_config() -> RunConfig:
    return load_config(REPO_ROOT / "configs" / "qwen3_0p6b_local.yaml")


@pytest.fixture(scope="session")
def qwen3_model_and_tokenizer(qwen3_run_config: RunConfig):
    return load_model_and_tokenizer(qwen3_run_config)


@pytest.fixture(scope="session")
def validation_context(qwen3_run_config: RunConfig, qwen3_model_and_tokenizer) -> ValidationContext:
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = build_adapter(model)
    return ValidationContext(config=qwen3_run_config, model=model, tokenizer=tokenizer, adapter=adapter)
