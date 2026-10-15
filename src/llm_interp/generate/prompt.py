from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    input_ids: torch.Tensor  # [1, seq_len], long, CPU — the source of truth
    offset_mapping: list[tuple[int, int]]  # character offsets into `text`, for hint-span lookup


def _extract_ids(result: Any) -> list[int]:
    """`apply_chat_template(tokenize=True, ...)` has returned a plain list in some
    transformers versions and a dict-like BatchEncoding in others (which is *not*
    a `dict` subclass — `isinstance(result, dict)` is False for it, and iterating
    it directly yields its keys, not `input_ids`); handle both via duck typing."""
    if hasattr(result, "keys"):
        return list(result["input_ids"])
    return list(result)


def render_and_tokenize(tokenizer, messages: list[dict], enable_thinking: bool = True) -> RenderedPrompt:
    """Render the prompt once (`enable_thinking=True` by default), tokenize that
    exact text once with `add_special_tokens=False`, and assert the result equals
    `apply_chat_template(..., tokenize=True)` directly.

    Never re-render through the chat template for anything downstream (capture,
    span-finding): Qwen3's template strips `<think>` content from earlier turns,
    and the token IDs returned here are the one source of truth from this point on.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]

    expected_ids = _extract_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    )
    if ids != expected_ids:
        raise ValueError(
            "render/tokenize mismatch: tokenizing the rendered text with "
            "add_special_tokens=False does not match apply_chat_template(tokenize=True) "
            f"directly.\n  from rendered text: {ids}\n  from apply_chat_template:  {expected_ids}"
        )

    return RenderedPrompt(
        text=text,
        input_ids=torch.tensor([ids], dtype=torch.long),
        offset_mapping=encoding["offset_mapping"],
    )
