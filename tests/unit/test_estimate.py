from pathlib import Path

from llm_interp.config import load_config
from llm_interp.estimate import default_tokens_per_trace, format_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_format_bytes_under_1kb():
    assert format_bytes(512) == "512.0 B"


def test_format_bytes_kb():
    assert format_bytes(2048) == "2.0 KB"


def test_format_bytes_mb():
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_format_bytes_gb():
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_format_bytes_zero():
    assert format_bytes(0) == "0.0 B"


def test_default_tokens_per_trace_is_cap_plus_nominal_prompt():
    config = load_config(REPO_ROOT / "configs" / "qwen3_0p6b_local.yaml")
    assert default_tokens_per_trace(config) == config.generation.max_new_tokens + 100
