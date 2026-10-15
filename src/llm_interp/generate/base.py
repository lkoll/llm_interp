from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

from ..config import GenerationConfig


@dataclass(frozen=True)
class GenerationResult:
    full_ids: torch.Tensor  # [total_len], prompt + generated, long, CPU
    prompt_len: int
    stop_reason: Literal["eos", "max_new_tokens"]
    truncated: bool
    seed: int

    @property
    def generated_ids(self) -> torch.Tensor:
        return self.full_ids[self.prompt_len :]


class Generator(ABC):
    """Takes pre-tokenized prompt IDs, never raw text — tokenization happens once,
    upstream, via `render_and_tokenize`. HFGenerator and a future VLLMGenerator
    share this interface so swapping one for the other is a config change."""

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> GenerationResult:
        """`input_ids`: [1, prompt_len], batch size 1."""
