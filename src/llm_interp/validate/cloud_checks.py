"""Addendum B: doctor additions for a RunPod pod. Gated on actually running on a
pod (RUNPOD_POD_ID set) rather than always-on, since the network-volume check in
particular would be a false failure against Phase 0's intentionally-local store
path -- same "skip as a trivial pass" convention as check (g)'s MLP-capture gate."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

from ..config import RunConfig
from ..estimate import estimate_bytes
from .base import CheckResult

RUNPOD_POD_ID_VAR = "RUNPOD_POD_ID"
DEFAULT_VOLUME_ROOT = "/workspace"
_THROUGHPUT_SAMPLE_BYTES = 64 * 1024 * 1024  # 64 MiB: enough for a stable MB/s reading, fast enough not to waste pod time


def _on_runpod_pod() -> bool:
    return bool(os.environ.get(RUNPOD_POD_ID_VAR))


def _skipped(name: str) -> CheckResult:
    return CheckResult(name, True, {"skipped": f"not running on a RunPod pod ({RUNPOD_POD_ID_VAR} not set)"})


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def check_cloud_gpu_report(config: RunConfig) -> CheckResult:
    """Report GPU, driver, CUDA version, bf16 support, and free VRAM. Informational
    rather than a correctness gate -- except a pod with no visible GPU at all,
    which is a real infra failure worth failing on."""
    name = "check_cloud_gpu_report"
    if not _on_runpod_pod():
        return _skipped(name)
    if not torch.cuda.is_available():
        return CheckResult(name, False, error="torch.cuda.is_available() is False on a RunPod GPU pod")

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    details = {
        "gpu_name": torch.cuda.get_device_name(device),
        "driver_version": _driver_version(),
        "cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "free_vram_gb": round(free_bytes / 1e9, 2),
        "total_vram_gb": round(total_bytes / 1e9, 2),
    }
    return CheckResult(name, True, details)


def check_cloud_network_volume(config: RunConfig, volume_root: str | None = None) -> CheckResult:
    """Fail if storage_root or HF_HOME isn't on the network volume -- that data
    vanishes with the pod otherwise. Defaults to /workspace (project_outline.md's
    Addendum B mount point); override via LLM_INTERP_VOLUME_ROOT if a pod ever
    mounts the volume elsewhere."""
    name = "check_cloud_network_volume"
    if not _on_runpod_pod():
        return _skipped(name)

    root = Path(volume_root or os.environ.get("LLM_INTERP_VOLUME_ROOT", DEFAULT_VOLUME_ROOT)).resolve()
    storage_root = Path(config.storage_root).resolve()
    hf_home = os.environ.get("HF_HOME")

    problems = []
    if not _is_under(storage_root, root):
        problems.append(f"storage_root {storage_root} is not under {root}")
    if hf_home is None:
        problems.append("HF_HOME is not set")
    elif not _is_under(Path(hf_home).resolve(), root):
        problems.append(f"HF_HOME {hf_home} is not under {root}")

    details = {"volume_root": str(root), "storage_root": str(storage_root), "hf_home": hf_home}
    if problems:
        return CheckResult(name, False, details, error="; ".join(problems))
    return CheckResult(name, True, details)


def check_cloud_storage_throughput(config: RunConfig) -> CheckResult:
    """Benchmark write/read throughput on the store path -- informational, not a
    pass/fail gate, since "slow" depends on the volume tier the user picked."""
    name = "check_cloud_storage_throughput"
    if not _on_runpod_pod():
        return _skipped(name)

    storage_root = Path(config.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    probe_path = storage_root / ".doctor_throughput_probe"
    payload = os.urandom(_THROUGHPUT_SAMPLE_BYTES)
    try:
        t0 = time.monotonic()
        with open(probe_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        write_seconds = time.monotonic() - t0

        t0 = time.monotonic()
        read_back = probe_path.read_bytes()
        read_seconds = time.monotonic() - t0
    finally:
        probe_path.unlink(missing_ok=True)

    if read_back != payload:
        return CheckResult(name, False, error="read-back of the throughput probe did not match what was written")

    mb = _THROUGHPUT_SAMPLE_BYTES / (1024 * 1024)
    details = {
        "sample_mb": round(mb, 1),
        "write_mb_per_s": round(mb / write_seconds, 1) if write_seconds > 0 else None,
        "read_mb_per_s": round(mb / read_seconds, 1) if read_seconds > 0 else None,
    }
    return CheckResult(name, True, details)


def check_cloud_estimate_vs_free_space(config: RunConfig, num_tokens: int) -> CheckResult:
    """Compare the run's estimated size (estimate.py's own byte accounting)
    against free space on the volume, so a capture doesn't run out of disk
    partway through."""
    name = "check_cloud_estimate_vs_free_space"
    if not _on_runpod_pod():
        return _skipped(name)

    storage_root = Path(config.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    result = estimate_bytes(config, num_tokens)
    free_bytes = shutil.disk_usage(storage_root).free

    details = {"projected_bytes": result.total_bytes, "free_bytes": free_bytes, "num_tokens": num_tokens}
    if result.total_bytes > free_bytes:
        return CheckResult(name, False, details, error="projected capture size exceeds free space on the volume")
    return CheckResult(name, True, details)
