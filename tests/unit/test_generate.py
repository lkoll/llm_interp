import pytest
import torch

from llm_interp.config import GenerationConfig
from llm_interp.generate import HFGenerator, VLLMGenerator
from llm_interp.generate.hf import _eos_token_ids, _stop_reason


def _generation_config(**overrides) -> GenerationConfig:
    base = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_new_tokens=8, seed=0)
    base.update(overrides)
    return GenerationConfig(**base)


def test_eos_token_ids_normalizes_int_and_list(tiny_qwen3_model):
    assert _eos_token_ids(tiny_qwen3_model) == {1}


def test_stop_reason_eos_when_last_token_is_eos():
    generated = torch.tensor([5, 6, 1])
    reason, truncated = _stop_reason(generated, {1})
    assert reason == "eos"
    assert truncated is False


def test_stop_reason_max_new_tokens_when_no_eos():
    generated = torch.tensor([5, 6, 7])
    reason, truncated = _stop_reason(generated, {1})
    assert reason == "max_new_tokens"
    assert truncated is True


def test_stop_reason_max_new_tokens_on_empty_generation():
    generated = torch.empty(0, dtype=torch.long)
    reason, truncated = _stop_reason(generated, {1})
    assert reason == "max_new_tokens"
    assert truncated is True


def test_hf_generator_output_starts_with_prompt(tiny_qwen3_model):
    generator = HFGenerator(tiny_qwen3_model)
    prompt_ids = torch.tensor([[2, 3, 4]])
    result = generator.generate(prompt_ids, _generation_config())

    torch.testing.assert_close(result.full_ids[: result.prompt_len], prompt_ids[0])
    assert result.prompt_len == 3
    assert result.full_ids.device.type == "cpu"
    assert result.seed == 0


def test_hf_generator_respects_max_new_tokens_cap(tiny_qwen3_model):
    generator = HFGenerator(tiny_qwen3_model)
    prompt_ids = torch.tensor([[2, 3, 4]])
    config = _generation_config(max_new_tokens=5, seed=7)
    result = generator.generate(prompt_ids, config)

    assert len(result.generated_ids) <= 5
    # the cap and the stop_reason/truncated flag must agree, whichever branch fired
    if result.stop_reason == "max_new_tokens":
        assert result.truncated is True
        assert len(result.generated_ids) == 5
    else:
        assert result.stop_reason == "eos"
        assert result.truncated is False


def test_hf_generator_is_deterministic_given_same_seed(tiny_qwen3_model):
    generator = HFGenerator(tiny_qwen3_model)
    prompt_ids = torch.tensor([[2, 3, 4]])
    config = _generation_config(max_new_tokens=6, seed=42)

    result_a = generator.generate(prompt_ids, config)
    result_b = generator.generate(prompt_ids, config)

    torch.testing.assert_close(result_a.full_ids, result_b.full_ids)
    assert result_a.stop_reason == result_b.stop_reason


def test_vllm_generator_is_an_unimplemented_stub():
    with pytest.raises(NotImplementedError):
        VLLMGenerator()
