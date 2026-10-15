import pytest

from llm_interp.generate import HFGenerator
from llm_interp.prompts import ArcEasyPromptSource, generate_pair

pytestmark = pytest.mark.gpu


def test_arc_easy_prompt_source_loads_30_items_with_letter_labels():
    items = ArcEasyPromptSource(n=30).load()
    assert len(items) == 30
    seen_ids = set()
    for item in items:
        assert item.pair_id not in seen_ids
        seen_ids.add(item.pair_id)
        assert item.gold in item.options
        assert all(label.isalpha() for label in item.options)
        assert len(item.options) >= 2


def test_generate_pair_real_model_runs_no_hint_then_hint(qwen3_model_and_tokenizer, qwen3_run_config):
    model, tokenizer = qwen3_model_and_tokenizer
    generator = HFGenerator(model)
    item = ArcEasyPromptSource(n=1).load()[0]

    traces = generate_pair(item, generator, tokenizer, qwen3_run_config.generation)

    assert [t.record.condition for t in traces] == ["no_hint", "hint"]
    no_hint_trace, hint_trace = traces

    assert no_hint_trace.record.hinted_option is None
    assert hint_trace.record.hinted_option is not None
    assert hint_trace.record.hinted_option != no_hint_trace.record.parsed_answer or (
        no_hint_trace.record.parsed_answer not in item.options
    )
    for trace in traces:
        assert trace.record.gold == item.gold
        assert trace.record.pair_id == item.pair_id
        assert trace.generation_result.prompt_len == trace.rendered_prompt.input_ids.shape[-1]
