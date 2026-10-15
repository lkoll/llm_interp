import torch

from ..config import GenerationConfig
from .base import GenerationResult, Generator


class VLLMGenerator(Generator):
    """Documented stub, not implemented in Phase 0.

    For Phase 1 (rented cloud GPU, larger models): vLLM pins its own torch version,
    incompatible with this project's, so it must run in a separate environment and
    write the same trace format rather than being imported here directly.

    Must accept pre-tokenized prompt IDs — exactly like `HFGenerator`, via the same
    `Generator.generate(input_ids, generation_config)` signature — never raw text,
    since tokenization happens once, upstream, via `render_and_tokenize`, and token
    IDs are this project's source of truth. Swapping `HFGenerator` for this one
    should be a config change, not a rewrite of anything that calls `Generator`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VLLMGenerator is a documented stub for Phase 1. Not implemented in Phase 0."
        )

    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> GenerationResult:
        raise NotImplementedError(
            "VLLMGenerator is a documented stub for Phase 1. Not implemented in Phase 0."
        )
