from dataclasses import dataclass
from typing import Literal

from ..config import GenerationConfig
from ..generate import GenerationResult, Generator, RenderedPrompt, render_and_tokenize
from .base import PromptItem
from .parse import ParsedAnswer, parse_answer
from .render import build_user_message


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    condition: Literal["no_hint", "hint"]
    hinted_option: str | None
    gold: str
    parsed_answer: str | None
    parse_status: Literal["ok", "not_found", "malformed", "multiple"]


@dataclass(frozen=True)
class PairTrace:
    """A `PairRecord` plus the rendered prompt and raw generation result it came
    from — carries the token IDs forward (the source of truth) for 0.7's capture
    stage to teacher-force-replay, rather than only the decoded, parsed summary.
    """

    record: PairRecord
    rendered_prompt: RenderedPrompt
    generation_result: GenerationResult


def choose_hinted_option(options: dict[str, str], no_hint_answer: str | None, gold: str) -> str:
    """An option letter != the model's no-hint answer; falls back to != gold if
    the no-hint answer wasn't one of the valid options (e.g. a parse failure)."""
    exclude = no_hint_answer if no_hint_answer in options else gold
    for label in options:
        if label != exclude:
            return label
    raise ValueError(f"no option available to hint that differs from {exclude!r}: {options}")


def parse_generated_answer(tokenizer, generated_ids: list[int]) -> ParsedAnswer:
    """Parse only the response (tokens after `</think>`), matching the `response`
    span capture uses — JSON the model writes while still reasoning must not count.
    No `</think>` (truncated generation) means no response, so "not_found"."""
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    try:
        close_idx = generated_ids.index(think_close_id)
    except ValueError:
        return ParsedAnswer(parsed_answer=None, parse_status="not_found")
    response_text = tokenizer.decode(generated_ids[close_idx + 1 :], skip_special_tokens=False)
    return parse_answer(response_text)


def _run_condition(
    item: PromptItem,
    condition: Literal["no_hint", "hint"],
    hinted_option: str | None,
    generator: Generator,
    tokenizer,
    generation_config: GenerationConfig,
) -> PairTrace:
    message = build_user_message(item, hinted_option)
    rendered = render_and_tokenize(
        tokenizer, [{"role": "user", "content": message}], enable_thinking=generation_config.enable_thinking
    )
    result = generator.generate(rendered.input_ids, generation_config)
    parsed: ParsedAnswer = parse_generated_answer(tokenizer, result.generated_ids.tolist())

    record = PairRecord(
        pair_id=item.pair_id,
        condition=condition,
        hinted_option=hinted_option,
        gold=item.gold,
        parsed_answer=parsed.parsed_answer,
        parse_status=parsed.parse_status,
    )
    return PairTrace(record=record, rendered_prompt=rendered, generation_result=result)


def generate_pair(
    item: PromptItem, generator: Generator, tokenizer, generation_config: GenerationConfig
) -> list[PairTrace]:
    """Run the no-hint condition first, then the hint condition with an option
    chosen to differ from the no-hint answer (or gold, on a parse failure)."""
    no_hint_trace = _run_condition(item, "no_hint", None, generator, tokenizer, generation_config)
    hinted_option = choose_hinted_option(item.options, no_hint_trace.record.parsed_answer, item.gold)
    hint_trace = _run_condition(item, "hint", hinted_option, generator, tokenizer, generation_config)
    return [no_hint_trace, hint_trace]
