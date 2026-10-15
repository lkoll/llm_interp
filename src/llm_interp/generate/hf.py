from typing import Literal

import torch
from torch import nn

from ..config import GenerationConfig
from .base import GenerationResult, Generator


def _eos_token_ids(model: nn.Module) -> set[int]:
    eos = model.config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def _stop_reason(generated_ids: torch.Tensor, eos_ids: set[int]) -> tuple[Literal["eos", "max_new_tokens"], bool]:
    """The only two ways `model.generate()` can stop here (no beam search, no
    custom stopping criteria): it emitted an eos token, or it hit the
    max_new_tokens cap first."""
    if generated_ids.numel() > 0 and generated_ids[-1].item() in eos_ids:
        return "eos", False
    return "max_new_tokens", True


class HFGenerator(Generator):
    """Generation via `model.generate()`. Needs only the model — not a tokenizer —
    since the interface takes and returns token IDs, never text."""

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> GenerationResult:
        torch.manual_seed(generation_config.seed)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[-1]
        eos_ids = _eos_token_ids(self.model)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=True,  # never greedy — causes repetition loops
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                min_p=generation_config.min_p,
                max_new_tokens=generation_config.max_new_tokens,
                pad_token_id=self.model.config.eos_token_id,
            )

        full_ids = output[0].detach().clone().to("cpu")
        stop_reason, truncated = _stop_reason(full_ids[prompt_len:], eos_ids)

        return GenerationResult(
            full_ids=full_ids,
            prompt_len=prompt_len,
            stop_reason=stop_reason,
            truncated=truncated,
            seed=generation_config.seed,
        )
