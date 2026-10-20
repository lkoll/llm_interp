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
pytestmark = pytest.mark.gpu


@pytest.fixture
def cloud_env(monkeypatch, tmp_path):
    """Simulates being on a RunPod pod: RUNPOD_POD_ID set, volume root + HF_HOME
    pointed at a real (tmp) directory so the checks exercise real file I/O
    instead of the skip path."""
    monkeypatch.setenv("RUNPOD_POD_ID", "test-pod")
    monkeypatch.setenv("LLM_INTERP_VOLUME_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return tmp_path


@pytest.fixture
def local_config():
    return load_config(REPO_ROOT / "configs" / "qwen3_0p6b_local.yaml")


def test_check_cloud_gpu_report_against_real_gpu(cloud_env, local_config):
    result = check_cloud_gpu_report(local_config)
    assert result.passed, result.error
    assert result.details["gpu_name"]
    assert result.details["cuda_version"]
    assert isinstance(result.details["bf16_supported"], bool)
    # >= 0, not > 0: this is informational (see the check's own docstring), and on
    # this machine's 6 GB card running after many other GPU tests in one process,
    # the caching allocator can legitimately leave ~0 bytes free -- that's a true
    # reading, not a bug.
    assert result.details["free_vram_gb"] >= 0
    assert result.details["total_vram_gb"] > 0


def test_check_cloud_network_volume_passes_when_under_root(cloud_env, local_config):
    config = local_config.model_copy(update={"storage_root": cloud_env / "runs" / "qwen3_0p6b_local"})
    result = check_cloud_network_volume(config)
    assert result.passed, result.error


def test_check_cloud_network_volume_fails_when_storage_root_outside_volume(cloud_env, local_config, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside_the_volume")
    config = local_config.model_copy(update={"storage_root": outside / "runs"})
    result = check_cloud_network_volume(config)
    assert not result.passed
    assert "is not under" in result.error


def test_check_cloud_storage_throughput_real_write_read(cloud_env, local_config):
    config = local_config.model_copy(update={"storage_root": cloud_env / "runs"})
    result = check_cloud_storage_throughput(config)
    assert result.passed, result.error
    assert result.details["write_mb_per_s"] > 0
    assert result.details["read_mb_per_s"] > 0
    assert not (config.storage_root / ".doctor_throughput_probe").exists()


def test_check_cloud_estimate_vs_free_space_real_disk(cloud_env, local_config):
    config = local_config.model_copy(update={"storage_root": cloud_env / "runs"})
    result = check_cloud_estimate_vs_free_space(config, num_tokens=1000)
    assert result.passed, result.error
    assert result.details["projected_bytes"] > 0
    assert result.details["free_bytes"] > result.details["projected_bytes"]
