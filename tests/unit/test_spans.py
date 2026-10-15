from llm_interp.generate.prompt import RenderedPrompt
from llm_interp.spans import find_answer_value_span, find_hint_span, find_response_span, find_think_spans

THINK_OPEN, THINK_CLOSE, EOS = 100, 101, 999


class _FakeTokenizer:
    """Maps each token id to a fixed string; `decode` just concatenates — lets the
    cumulative-decode mapping logic in find_answer_value_span be tested without a
    real BPE tokenizer."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.vocab[i] for i in ids)


def test_find_think_spans_normal_case():
    # [prompt..., THINK_OPEN, 1, 2, THINK_CLOSE, 3, 4, EOS]
    full_ids = [10, 11, THINK_OPEN, 1, 2, THINK_CLOSE, 3, 4, EOS]
    spans = find_think_spans(full_ids, THINK_OPEN, THINK_CLOSE)
    assert spans is not None
    assert spans["think_open"].start == 2 and spans["think_open"].end == 3
    assert spans["think"].start == 3 and spans["think"].end == 5
    assert spans["think_close"].start == 5 and spans["think_close"].end == 6


def test_find_think_spans_none_when_open_missing():
    full_ids = [10, 11, 1, 2, THINK_CLOSE, 3, 4, EOS]
    assert find_think_spans(full_ids, THINK_OPEN, THINK_CLOSE) is None


def test_find_think_spans_none_when_close_missing_truncated():
    full_ids = [10, 11, THINK_OPEN, 1, 2, 3, 4]
    assert find_think_spans(full_ids, THINK_OPEN, THINK_CLOSE) is None


def test_find_response_span_ends_at_eos():
    full_ids = [10, 11, THINK_OPEN, 1, THINK_CLOSE, 3, 4, EOS, 0]
    think_close_end = 5  # position right after </think>
    span = find_response_span(full_ids, think_close_end, {EOS})
    assert span.start == 5
    assert span.end == 7  # excludes EOS at index 7


def test_find_response_span_extends_to_end_when_no_eos_truncated():
    full_ids = [10, 11, THINK_OPEN, 1, THINK_CLOSE, 3, 4]
    span = find_response_span(full_ids, 5, {EOS})
    assert span.start == 5
    assert span.end == 7  # len(full_ids), no eos found


def _rendered_prompt(text: str, offset_mapping: list[tuple[int, int]]) -> RenderedPrompt:
    import torch

    return RenderedPrompt(text=text, input_ids=torch.zeros((1, len(offset_mapping)), dtype=torch.long), offset_mapping=offset_mapping)


def test_find_hint_span_none_when_no_hint():
    rendered = _rendered_prompt("hello", [(0, 5)])
    assert find_hint_span(rendered, None) is None


def test_find_hint_span_locates_overlapping_tokens():
    text = "Question text. A Stanford professor thinks the answer is B. More text."
    hint_text = "A Stanford professor thinks the answer is B."
    char_start = text.index(hint_text)
    char_end = char_start + len(hint_text)
    # tokens: before-hint, [overlapping hint tokens...], after-hint
    offsets = [
        (0, char_start),  # token 0: everything before the hint
        (char_start, char_start + 10),  # token 1: inside hint
        (char_start + 10, char_end - 1),  # token 2: inside hint
        (char_end - 1, char_end + 5),  # token 3: overlaps the end of the hint
        (char_end + 5, len(text)),  # token 4: after the hint
    ]
    rendered = _rendered_prompt(text, offsets)
    span = find_hint_span(rendered, "B")
    assert span.start == 1
    assert span.end == 4  # tokens 1, 2, 3 overlap the hint; half-open end


def test_find_answer_value_span_simple():
    # response text (after decode): 'reasoning.\n{"answer": "B"}'
    # token ids map 1:1 to short fragments for a deterministic, inspectable mapping
    vocab = {
        1: "reasoning.\n{",
        2: '"answer"',
        3: ": ",
        4: '"B"',
        5: "}",
    }
    tok = _FakeTokenizer(vocab)
    full_ids = [0, 0, 1, 2, 3, 4, 5]  # response starts at index 2
    response_span_start, response_span_end = 2, 7
    from llm_interp.spans import Span

    span = find_answer_value_span(tok, full_ids, Span(response_span_start, response_span_end))
    assert span is not None
    # the answer letter "B" lives entirely inside token id 4 ('"B"')
    assert span.start == 5  # absolute position of token 4 in full_ids
    assert span.end == 6


def test_find_answer_value_span_none_when_not_found():
    vocab = {1: "no json here"}
    tok = _FakeTokenizer(vocab)
    full_ids = [0, 0, 1]
    from llm_interp.spans import Span

    assert find_answer_value_span(tok, full_ids, Span(2, 3)) is None
