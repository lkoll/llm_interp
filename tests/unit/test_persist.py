import torch

from llm_interp.generate import GenerationResult, RenderedPrompt
from llm_interp.prompts import PairRecord, PairTrace, load_pair_traces, save_pair_traces


def _sample_trace(pair_id: str, condition: str, hinted_option: str | None) -> PairTrace:
    record = PairRecord(
        pair_id=pair_id,
        condition=condition,
        hinted_option=hinted_option,
        gold="A",
        parsed_answer="A",
        parse_status="ok",
    )
    rendered_prompt = RenderedPrompt(
        text="hello world",
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        offset_mapping=[(0, 5), (5, 6), (6, 11)],
    )
    generation_result = GenerationResult(
        full_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        prompt_len=3,
        stop_reason="eos",
        truncated=False,
        seed=42,
    )
    return PairTrace(record=record, rendered_prompt=rendered_prompt, generation_result=generation_result)


def test_save_and_load_pair_traces_round_trip(tmp_path):
    traces = [_sample_trace("p1", "no_hint", None), _sample_trace("p1", "hint", "B")]
    path = tmp_path / "generated_traces.json"

    save_pair_traces(path, traces)
    loaded = load_pair_traces(path)

    assert len(loaded) == 2
    for original, round_tripped in zip(traces, loaded):
        assert round_tripped.record == original.record
        assert round_tripped.rendered_prompt.text == original.rendered_prompt.text
        assert round_tripped.rendered_prompt.offset_mapping == original.rendered_prompt.offset_mapping
        torch.testing.assert_close(round_tripped.rendered_prompt.input_ids, original.rendered_prompt.input_ids)
        torch.testing.assert_close(round_tripped.generation_result.full_ids, original.generation_result.full_ids)
        assert round_tripped.generation_result.prompt_len == original.generation_result.prompt_len
        assert round_tripped.generation_result.stop_reason == original.generation_result.stop_reason
        assert round_tripped.generation_result.truncated == original.generation_result.truncated
        assert round_tripped.generation_result.seed == original.generation_result.seed
        torch.testing.assert_close(round_tripped.generation_result.generated_ids, original.generation_result.generated_ids)


def test_save_pair_traces_is_human_readable_json(tmp_path):
    import json

    traces = [_sample_trace("p1", "no_hint", None)]
    path = tmp_path / "generated_traces.json"
    save_pair_traces(path, traces)

    data = json.loads(path.read_text())
    assert data[0]["record"]["pair_id"] == "p1"
    assert data[0]["generation_result"]["full_ids"] == [1, 2, 3, 4, 5]
