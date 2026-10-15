from pathlib import Path

import yaml

from .schema import RunConfig


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
