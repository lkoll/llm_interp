from dataclasses import dataclass, field
from typing import Any

import torch

from ..adapters.base import ArchAdapter
from ..config import RunConfig


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


# Tight in fp32; looser in bf16. For checks that compare the *same* forward pass
# against a recomputation of it (a, c, d, f) — true numerical identity is expected.
TOLERANCES: dict[str, Tolerance] = {
    "float32": Tolerance(atol=1e-4, rtol=1e-4),
    "bfloat16": Tolerance(atol=2e-2, rtol=2e-2),
}

# Check (b) compares two genuinely different computational paths (KV-cached
# incremental decode vs. one teacher-forced pass) — CLAUDE.md: "cached decoding
# and full forward passes legitimately differ," even in fp32. Relative error is
# the wrong gate here (it blows up on near-zero activations despite the absolute
# difference being tiny), so this is an absolute-error-only threshold; cosine
# similarity and argmax agreement are the other two gates actually used.
CACHE_VS_TEACHER_FORCED_ATOL: dict[str, float] = {
    "float32": 1e-2,
    "bfloat16": 5e-2,
}
MIN_COSINE_SIMILARITY = 0.999


def get_tolerance(dtype: str) -> Tolerance:
    return TOLERANCES[dtype]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ValidationContext:
    config: RunConfig
    model: torch.nn.Module
    tokenizer: Any
    adapter: ArchAdapter

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def tolerance(self) -> Tolerance:
        return get_tolerance(self.config.model.dtype)
