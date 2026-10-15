import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

# Deliberately head_dim != hidden_size // num_attention_heads (4 != 8), mirroring the
# real Qwen3-0.6B gotcha (head_dim 128 != hidden_size 1024 // num_attention_heads 16 == 64).
TINY_QWEN3_KWARGS = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000},
    eos_token_id=1,
    pad_token_id=1,
)


@pytest.fixture
def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(**TINY_QWEN3_KWARGS)


@pytest.fixture
def tiny_qwen3_model(tiny_qwen3_config: Qwen3Config) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_qwen3_config)
    model.eval()
    return model
