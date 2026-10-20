from pathlib import Path

import pytest

from llm_interp.config import load_config
from llm_interp.validate.cloud_checks import (
    check_cloud_estimate_vs_free_space,
    check_cloud_gpu_report,
    check_cloud_network_volume,
    check_cloud_storage_throughput,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def local_config():
    return load_config(REPO_ROOT / "configs" / "qwen3_0p6b_local.yaml")


def test_all_cloud_checks_skip_when_not_on_a_pod(monkeypatch, local_config):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

    results = [
        check_cloud_gpu_report(local_config),
        check_cloud_network_volume(local_config),
        check_cloud_storage_throughput(local_config),
        check_cloud_estimate_vs_free_space(local_config, num_tokens=1000),
    ]
    assert all(r.passed for r in results)
    assert all(r.details.get("skipped") for r in results)
