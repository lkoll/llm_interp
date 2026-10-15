import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import torch

from ..config import RunConfig

PINNED_PACKAGES = ["torch", "transformers", "zarr", "pyarrow", "duckdb", "pydantic", "datasets", "pyyaml"]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_info() -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        name, driver = (s.strip() for s in result.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "driver_version": driver}
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return {"name": None, "driver_version": None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in PINNED_PACKAGES:
        try:
            versions[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(config: RunConfig, tokenizer) -> dict:
    chat_template = tokenizer.chat_template or ""
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    chat_template_hash = hashlib.sha256(chat_template.encode()).hexdigest()

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "cuda_version": torch.version.cuda,
        "gpu": _gpu_info(),
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": chat_template_hash,
        "config": config.model_dump(mode="json"),
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2))
