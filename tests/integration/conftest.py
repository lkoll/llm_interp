import os

# Must be set before the first CUDA context init for deterministic cuBLAS algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_interp.config import RunConfig, load_config

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
    cfg = qwen3_run_config
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.float32,
        attn_implementation=cfg.model.attn_implementation,
    )
    model = model.to(cfg.model.device)
    model.eval()
    return model, tokenizer
