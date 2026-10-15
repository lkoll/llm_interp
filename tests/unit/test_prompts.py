import pytest

from llm_interp.prompts import PromptItem, build_user_message, choose_hinted_option, parse_answer
from llm_interp.prompts.render import HINT_TEMPLATE


def test_parse_answer_simple():
    result = parse_answer('{"answer": "C"}')
    assert result.parsed_answer == "C"
    assert result.parse_status == "ok"


def test_parse_answer_with_reasoning_prefix_and_suffix():
    text = 'Let me think... the sky is blue.\n{"answer": "B"}\nThat is my final answer.'
    result = parse_answer(text)
    assert result.parsed_answer == "B"
    assert result.parse_status == "ok"


def test_parse_answer_not_found_when_no_json():
    result = parse_answer("I think the answer is B, but I will not say so in JSON.")
    assert result.parsed_answer is None
    assert result.parse_status == "not_found"


def test_parse_answer_not_found_ignores_invalid_json_braces():
    result = parse_answer("{this is not valid json}")
    assert result.parsed_answer is None
    assert result.parse_status == "not_found"


def test_parse_answer_malformed_when_answer_value_not_string():
    result = parse_answer('{"answer": 3}')
    assert result.parsed_answer is None
    assert result.parse_status == "malformed"


def test_parse_answer_malformed_when_answer_value_empty_string():
    result = parse_answer('{"answer": ""}')
    assert result.parsed_answer is None
    assert result.parse_status == "malformed"


def test_parse_answer_multiple_occurrences_same_value_is_ok():
    text = 'First I said {"answer": "A"} but then confirmed {"answer": "A"}.'
    result = parse_answer(text)
    assert result.parsed_answer == "A"
    assert result.parse_status == "ok"


def test_parse_answer_multiple_occurrences_different_values():
    text = 'First I thought {"answer": "A"} but changed my mind: {"answer": "D"}.'
    result = parse_answer(text)
    assert result.parsed_answer == "D"  # takes the last
    assert result.parse_status == "multiple"


def test_parse_answer_handles_escaped_quotes_in_string():
    text = '{"note": "she said \\"hi\\"", "answer": "C"}'
    result = parse_answer(text)
    assert result.parsed_answer == "C"
    assert result.parse_status == "ok"


def test_parse_answer_handles_nested_object():
    text = '{"meta": {"confidence": "high"}, "answer": "C"}'
    result = parse_answer(text)
    assert result.parsed_answer == "C"
    assert result.parse_status == "ok"


def _item() -> PromptItem:
    return PromptItem(
        pair_id="test_1",
        question="What color is the sky?",
        options={"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        gold="B",
    )


def test_build_user_message_without_hint_has_no_hint_line():
    message = build_user_message(_item(), None)
    assert "Stanford professor" not in message
    assert "A. Red" in message
    assert "B. Blue" in message


def test_build_user_message_with_hint_includes_hint_line():
    message = build_user_message(_item(), "D")
    assert HINT_TEMPLATE.format(option="D") in message


def test_choose_hinted_option_excludes_no_hint_answer():
    item = _item()
    chosen = choose_hinted_option(item.options, no_hint_answer="B", gold="B")
    assert chosen != "B"
    assert chosen in item.options


def test_choose_hinted_option_falls_back_to_gold_when_no_hint_answer_invalid():
    item = _item()
    # no_hint_answer "Z" isn't a real option (e.g. a parse failure) -> fall back to excluding gold
    chosen = choose_hinted_option(item.options, no_hint_answer="Z", gold="B")
    assert chosen != "B"
    assert chosen in item.options


def test_choose_hinted_option_falls_back_to_gold_when_no_hint_answer_is_none():
    item = _item()
    chosen = choose_hinted_option(item.options, no_hint_answer=None, gold="B")
    assert chosen != "B"


def test_choose_hinted_option_raises_when_no_alternative_exists():
    with pytest.raises(ValueError):
        choose_hinted_option({"A": "only option"}, no_hint_answer="A", gold="A")
