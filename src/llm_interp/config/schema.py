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


class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    layers: list[int]
    capture_final_residual: bool = True
    top_k_logprobs: int = Field(default=20, gt=0)
    capture_qk_pre_rope: bool = False

    @field_validator("layers")
    @classmethod
    def _layers_valid(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError(f"layers must not contain duplicates, got {v}")
        if any(layer < 0 for layer in v):
            raise ValueError(f"layers must be non-negative, got {v}")
        return sorted(v)


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
