from dataclasses import dataclass

from .generate.prompt import RenderedPrompt
from .prompts.parse import parse_answer
from .prompts.render import HINT_TEMPLATE


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # half-open [start, end), token positions in the full sequence


def find_think_spans(full_ids: list[int], think_open_id: int, think_close_id: int) -> dict[str, Span] | None:
    """`think`: tokens strictly between `<think>` and `</think>`, plus `think_open`/
    `think_close` as one-token spans. None if `</think>` never appears (truncated
    generation) — caller must flag the trace and skip the `response` span too."""
    try:
        open_idx = full_ids.index(think_open_id)
    except ValueError:
        return None
    try:
        close_idx = full_ids.index(think_close_id, open_idx + 1)
    except ValueError:
        return None
    return {
        "think_open": Span(open_idx, open_idx + 1),
        "think": Span(open_idx + 1, close_idx),
        "think_close": Span(close_idx, close_idx + 1),
    }


def find_response_span(full_ids: list[int], think_close_end: int, eos_ids: set[int]) -> Span:
    """First token after `</think>` up to, not including, eos. If no eos is found
    (truncated mid-response), extends to the end of the sequence. Whitespace
    tokens are kept; trim at analysis time."""
    start = think_close_end
    end = len(full_ids)
    for i in range(start, len(full_ids)):
        if full_ids[i] in eos_ids:
            end = i
            break
    return Span(start, end)


def find_hint_span(rendered_prompt: RenderedPrompt, hinted_option: str | None) -> Span | None:
    """Located via the tokenizer's offset mapping on the rendered prompt — unlike
    the response, the prompt's text is ours, so real offsets are available
    directly (no need for incremental re-decoding)."""
    if hinted_option is None:
        return None
    hint_text = HINT_TEMPLATE.format(option=hinted_option)
    char_start = rendered_prompt.text.index(hint_text)
    char_end = char_start + len(hint_text)

    token_start: int | None = None
    token_end: int | None = None
    for token_idx, (tok_start, tok_end) in enumerate(rendered_prompt.offset_mapping):
        if tok_start == tok_end:
            continue
        if tok_start < char_end and tok_end > char_start:
            if token_start is None:
                token_start = token_idx
            token_end = token_idx + 1
    if token_start is None or token_end is None:
        return None
    return Span(token_start, token_end)


def _cumulative_decoded_lengths(tokenizer, token_ids: list[int]) -> list[int]:
    """cum[0] = 0; cum[i] = len(decode(token_ids[:i])) for i in 1..len(token_ids).
    Used to map a character offset in decoded text back to a token position
    without re-tokenizing — generated tokens have no offset_mapping of their own."""
    cum = [0]
    for i in range(1, len(token_ids) + 1):
        cum.append(len(tokenizer.decode(token_ids[:i], skip_special_tokens=False)))
    return cum


def find_answer_value_span(tokenizer, full_ids: list[int], response_span: Span) -> Span | None:
    """Sub-span (absolute positions in `full_ids`) for the token(s) carrying the
    parsed answer letter, from the LAST valid JSON "answer" in the response. Maps
    the character offset to token positions by incrementally decoding prefixes of
    the stored response token IDs — never by re-tokenizing the decoded text.
    None if no valid JSON answer was found in the response (parse_status would be
    "not_found" or "malformed")."""
    response_ids = full_ids[response_span.start : response_span.end]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    parsed = parse_answer(response_text)
    if parsed.parsed_answer is None:
        return None

    needle = f'"{parsed.parsed_answer}"'
    quoted_start = response_text.rindex(needle)
    char_start = quoted_start + 1  # skip the opening quote
    char_end = char_start + len(parsed.parsed_answer)

    cum = _cumulative_decoded_lengths(tokenizer, response_ids)
    start_tok = next(i for i in range(len(response_ids)) if cum[i] <= char_start < cum[i + 1])
    end_tok = next(i for i in range(len(response_ids)) if cum[i + 1] >= char_end)
    return Span(response_span.start + start_tok, response_span.start + end_tok + 1)
