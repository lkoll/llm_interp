import copy

import pytest
import torch
import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from llm_interp.adapters import Qwen3Adapter


def test_dims_read_from_config(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    cfg = tiny_qwen3_model.config
    assert adapter.num_layers == cfg.num_hidden_layers == 2
    assert adapter.hidden_size == cfg.hidden_size == 32
    assert adapter.num_attention_heads == cfg.num_attention_heads == 4
    assert adapter.num_key_value_heads == cfg.num_key_value_heads == 2
    assert adapter.head_dim == cfg.head_dim == 16
    assert adapter.intermediate_size == cfg.intermediate_size == 64
    # the gotcha this fixture is designed to exercise
    assert adapter.head_dim != adapter.hidden_size // adapter.num_attention_heads


def test_module_getters_return_real_submodules(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    assert adapter.embeddings() is tiny_qwen3_model.model.embed_tokens
    assert adapter.final_norm() is tiny_qwen3_model.model.norm
    assert adapter.lm_head() is tiny_qwen3_model.lm_head
    for layer_idx in range(adapter.num_layers):
        layer = tiny_qwen3_model.model.layers[layer_idx]
        assert adapter.layer(layer_idx) is layer
        assert adapter.attn_input_norm(layer_idx) is layer.input_layernorm
        assert adapter.q_proj(layer_idx) is layer.self_attn.q_proj
        assert adapter.k_proj(layer_idx) is layer.self_attn.k_proj
        assert adapter.v_proj(layer_idx) is layer.self_attn.v_proj
        assert adapter.q_norm(layer_idx) is layer.self_attn.q_norm
        assert adapter.k_norm(layer_idx) is layer.self_attn.k_norm
        assert adapter.rotary_emb(layer_idx) is tiny_qwen3_model.model.rotary_emb


def test_attention_type_is_full_for_all_layers(tiny_qwen3_model):
    # no sliding window configured for this fixture (mirrors Qwen3-0.6B)
    adapter = Qwen3Adapter(tiny_qwen3_model)
    for layer_idx in range(adapter.num_layers):
        assert adapter.attention_type(layer_idx) == "full_attention"


def test_layer_idx_out_of_range_raises(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    import pytest

    with pytest.raises(IndexError):
        adapter.layer(adapter.num_layers)
    with pytest.raises(IndexError):
        adapter.layer(-1)


def test_reshape_to_heads_matches_attention_internal_shape(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    batch, seq_len = 1, 5
    x = torch.randn(batch, seq_len, adapter.num_attention_heads * adapter.head_dim)
    reshaped = adapter.reshape_to_heads(x, adapter.num_attention_heads)
    assert reshaped.shape == (batch, adapter.num_attention_heads, seq_len, adapter.head_dim)


def test_apply_qk_norm_and_rotary_match_real_forward(tiny_qwen3_model):
    """adapter.reshape_to_heads -> apply_qk_norm -> apply_rotary must reproduce,
    bit-for-bit, what Qwen3Attention.forward computes internally. Spies on the
    model's own q_proj/k_proj outputs and its module-level apply_rotary_pos_emb
    call to get ground truth, since there's no other hook point for the
    post-norm/pre-rotary intermediate."""
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 1
    batch, seq_len = 1, 5

    torch.manual_seed(1)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (batch, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    proj_out: dict[str, torch.Tensor] = {}
    h_q = adapter.q_proj(layer_idx).register_forward_hook(
        lambda module, args, output: proj_out.__setitem__("q", output.detach().clone())
    )
    h_k = adapter.k_proj(layer_idx).register_forward_hook(
        lambda module, args, output: proj_out.__setitem__("k", output.detach().clone())
    )

    captured: dict[str, torch.Tensor] = {}
    original_rotary_fn = qwen3_modeling.apply_rotary_pos_emb

    def spy(q, k, cos, sin, unsqueeze_dim=1):
        captured["q_pre_rotary"] = q.detach().clone()
        captured["k_pre_rotary"] = k.detach().clone()
        q_out, k_out = original_rotary_fn(q, k, cos, sin, unsqueeze_dim)
        captured["q_post_rotary"] = q_out.detach().clone()
        captured["k_post_rotary"] = k_out.detach().clone()
        return q_out, k_out

    qwen3_modeling.apply_rotary_pos_emb = spy
    try:
        with torch.no_grad():
            tiny_qwen3_model(input_ids=ids, position_ids=position_ids)
    finally:
        qwen3_modeling.apply_rotary_pos_emb = original_rotary_fn
        h_q.remove()
        h_k.remove()

    q_reshaped = adapter.reshape_to_heads(proj_out["q"], adapter.num_attention_heads)
    k_reshaped = adapter.reshape_to_heads(proj_out["k"], adapter.num_key_value_heads)
    q_normed, k_normed = adapter.apply_qk_norm(q_reshaped, k_reshaped, layer_idx)

    torch.testing.assert_close(q_normed, captured["q_pre_rotary"])
    torch.testing.assert_close(k_normed, captured["k_pre_rotary"])

    q_rot, k_rot = adapter.apply_rotary(q_normed, k_normed, position_ids, layer_idx)
    torch.testing.assert_close(q_rot, captured["q_post_rotary"])
    torch.testing.assert_close(k_rot, captured["k_post_rotary"])


def test_mlp_module_getters_return_real_submodules(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    for layer_idx in range(adapter.num_layers):
        mlp = tiny_qwen3_model.model.layers[layer_idx].mlp
        assert adapter.mlp(layer_idx) is mlp
        assert adapter.mlp_gate_proj(layer_idx) is mlp.gate_proj
        assert adapter.mlp_up_proj(layer_idx) is mlp.up_proj
        assert adapter.mlp_down_proj(layer_idx) is mlp.down_proj


def test_mlp_activation_fn_is_the_real_act_fn(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    assert adapter.mlp_activation_fn() is tiny_qwen3_model.model.layers[0].mlp.act_fn

    x = torch.randn(5)
    torch.testing.assert_close(adapter.mlp_activation_fn()(x), torch.nn.functional.silu(x))


def test_mlp_activation_negative_min_for_silu(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    negative_min = adapter.mlp_activation_negative_min
    assert negative_min == pytest.approx(-0.2784645427610738)

    # SiLU on the negative side never goes below its analytical infimum
    x = torch.linspace(-50, 0, 10_000)
    silu_values = torch.nn.functional.silu(x)
    assert silu_values.min().item() >= negative_min - 1e-6


def test_mlp_activation_negative_min_raises_for_unknown_activation(tiny_qwen3_config):
    cfg = copy.deepcopy(tiny_qwen3_config)
    cfg.hidden_act = "relu"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    adapter = Qwen3Adapter(model)
    with pytest.raises(NotImplementedError):
        adapter.mlp_activation_negative_min
