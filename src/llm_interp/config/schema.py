from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    revision: str
    dtype: Literal["float32", "bfloat16"]
    device: str
    attn_implementation: Literal["sdpa", "eager"]

    @field_validator("device")
    @classmethod
    def _device_known(cls, v: str) -> str:
        if v != "cpu" and not v.startswith("cuda"):
            raise ValueError(f"device must be 'cpu' or start with 'cuda', got {v!r}")
        return v


def _validate_layers(v: list[int]) -> list[int]:
    if len(set(v)) != len(v):
        raise ValueError(f"layers must not contain duplicates, got {v}")
    if any(layer < 0 for layer in v):
        raise ValueError(f"layers must be non-negative, got {v}")
    return sorted(v)


class MlpCaptureConfig(BaseModel):
    """Addendum A: native MLP neuron capture. `dense_layers` and `topk_layers`
    are independent and may overlap (a layer can get both representations) —
    e.g. the cloud config's "topk on all layers plus dense on 4"."""

    model_config = ConfigDict(frozen=True)

    dense_layers: list[int] = Field(default_factory=list)
    topk_layers: list[int] = Field(default_factory=list)
    top_k: int = Field(default=64, gt=0)
    # Additionally capture dense mlp_gate/mlp_up/mlp_out, for check (g) only —
    # not for the main capture run (mirrors capture_qk_pre_rope's "validation
    # runs only" below).
    capture_validation_extras: bool = False

    @field_validator("dense_layers", "topk_layers")
    @classmethod
    def _layers_valid(cls, v: list[int]) -> list[int]:
        return _validate_layers(v)


class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    layers: list[int]
    capture_final_residual: bool = True
    top_k_logprobs: int = Field(default=20, gt=0)
    capture_qk_pre_rope: bool = False
    mlp: MlpCaptureConfig | None = None

    @field_validator("layers")
    @classmethod
    def _layers_valid(cls, v: list[int]) -> list[int]:
        return _validate_layers(v)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(gt=0)
    min_p: float = Field(ge=0.0, lt=1.0)
    max_new_tokens: int = Field(gt=0)
    seed: int = Field(ge=0)
    enable_thinking: bool = True


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_name: str
    model: ModelConfig
    capture: CaptureConfig
    generation: GenerationConfig
    storage_root: Path
