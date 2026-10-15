import pytest
import torch

from llm_interp.generate import HFGenerator, render_and_tokenize
from llm_interp.generate.prompt import _extract_ids

pytestmark = pytest.mark.gpu


def test_render_and_tokenize_matches_apply_chat_template_directly(qwen3_model_and_tokenizer):
    _, tokenizer = qwen3_model_and_tokenizer
    messages = [{"role": "user", "content": "What color is the sky?"}]

    rendered = render_and_tokenize(tokenizer, messages, enable_thinking=True)

    expected_ids = _extract_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    )
    assert rendered.input_ids.tolist() == [expected_ids]


def test_render_and_tokenize_enable_thinking_false_inserts_empty_think_block(qwen3_model_and_tokenizer):
    _, tokenizer = qwen3_model_and_tokenizer
    messages = [{"role": "user", "content": "What color is the sky?"}]

    rendered = render_and_tokenize(tokenizer, messages, enable_thinking=False)

    assert "<think>\n\n</think>" in rendered.text


def test_render_and_tokenize_enable_thinking_true_leaves_think_open(qwen3_model_and_tokenizer):
    _, tokenizer = qwen3_model_and_tokenizer
    messages = [{"role": "user", "content": "What color is the sky?"}]

    rendered = render_and_tokenize(tokenizer, messages, enable_thinking=True)

    assert "<think>" not in rendered.text
    assert rendered.text.endswith("<|im_start|>assistant\n")


def test_hf_generator_real_model_thinks_then_answers_then_eos(qwen3_model_and_tokenizer, qwen3_run_config):
    model, tokenizer = qwen3_model_and_tokenizer
    gen_cfg = qwen3_run_config.generation

    messages = [{"role": "user", "content": 'What color is the sky? Answer with JSON like {"answer": "blue"}.'}]
    rendered = render_and_tokenize(tokenizer, messages, enable_thinking=gen_cfg.enable_thinking)

    result = HFGenerator(model).generate(rendered.input_ids, gen_cfg)

    assert result.prompt_len == rendered.input_ids.shape[-1]
    assert len(result.generated_ids) <= gen_cfg.max_new_tokens
    assert result.seed == gen_cfg.seed

    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    generated = result.generated_ids.tolist()
    assert think_open_id in generated
    assert think_close_id in generated
    assert generated.index(think_open_id) < generated.index(think_close_id)

    if result.stop_reason == "eos":
        eos_ids = model.config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        assert generated[-1] in eos_ids
    else:
        assert result.truncated is True
        assert len(generated) == gen_cfg.max_new_tokens
