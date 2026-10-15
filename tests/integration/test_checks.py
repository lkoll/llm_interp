import pytest

from llm_interp.validate import (
    check_a_logits_from_final_residual,
    check_b_kv_cache_vs_teacher_forced,
    check_c_recomputed_qk,
    check_d_attention_probs_vs_eager,
    check_e_storage_round_trip,
    check_f_add_vector,
)

pytestmark = pytest.mark.gpu

VALIDATION_TEXT = "The quick brown fox jumps over the lazy dog, and the five boxing wizards jump quickly."


@pytest.fixture(scope="module")
def input_ids(qwen3_model_and_tokenizer):
    _, tokenizer = qwen3_model_and_tokenizer
    return tokenizer(VALIDATION_TEXT, return_tensors="pt").input_ids


def test_check_a_logits_from_final_residual(validation_context, input_ids):
    result = check_a_logits_from_final_residual(validation_context, input_ids)
    assert result.passed, result.details or result.error


def test_check_b_kv_cache_vs_teacher_forced(validation_context, input_ids):
    result = check_b_kv_cache_vs_teacher_forced(validation_context, input_ids[0])
    assert result.passed, result.details or result.error


def test_check_c_recomputed_qk_matches_hooked_qk_norm(validation_context, input_ids):
    result = check_c_recomputed_qk(validation_context, input_ids)
    assert result.passed, result.details or result.error


def test_check_d_attention_probs_vs_eager(validation_context, input_ids):
    result = check_d_attention_probs_vs_eager(validation_context, input_ids)
    assert result.passed, result.details or result.error
    # must restore the configured attn_implementation afterward
    assert validation_context.model.config._attn_implementation == validation_context.config.model.attn_implementation


def test_check_e_storage_round_trip():
    result = check_e_storage_round_trip()
    assert result.passed, result.details or result.error


def test_check_f_add_vector(validation_context, input_ids):
    result = check_f_add_vector(validation_context, input_ids)
    assert result.passed, result.details or result.error
