import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import ArchAdapter, Qwen3Adapter
from .config import RunConfig

_TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_model_and_tokenizer(config: RunConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=_TORCH_DTYPES[config.model.dtype],
        attn_implementation=config.model.attn_implementation,
    )
    model = model.to(config.model.device)
    model.eval()
    return model, tokenizer


def build_adapter(model: torch.nn.Module) -> ArchAdapter:
    # Only Qwen3Adapter exists in Phase 0; model-id-based adapter selection is a
    # future-adapter (Gemma 3, Qwen3.6) concern, not needed yet.
    return Qwen3Adapter(model)
