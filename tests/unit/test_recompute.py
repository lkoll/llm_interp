import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.hooks import HookManager
from llm_interp.recompute import (
    recompute_attention,
    recompute_logits,
    recompute_qk,
    recompute_qk_pre_rope,
    recompute_v,
    repeat_kv,
)


def test_attention_scaling_matches_real_submodule(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    for layer_idx in range(adapter.num_layers):
        real = tiny_qwen3_model.model.layers[layer_idx].self_attn.scaling
        assert adapter.attention_scaling(layer_idx) == real


def test_repeat_kv_gqa_mapping():
    # 2 kv heads, n_rep=3 -> 6 query-facing heads; head h must read kv head h // 3.
    batch, num_kv_heads, seq_len, head_dim, n_rep = 1, 2, 4, 3, 3
    kv = torch.arange(num_kv_heads, dtype=torch.float32).view(1, num_kv_heads, 1, 1).expand(
        batch, num_kv_heads, seq_len, head_dim
    )
    repeated = repeat_kv(kv, n_rep)
    assert repeated.shape == (batch, num_kv_heads * n_rep, seq_len, head_dim)
    for h in range(num_kv_heads * n_rep):
        expected_kv_head = h // n_rep
        assert torch.all(repeated[:, h] == expected_kv_head)


def test_repeat_kv_identity_when_n_rep_one():
    x = torch.randn(1, 4, 5, 3)
    torch.testing.assert_close(repeat_kv(x, 1), x)


def _capture_resid_pre_and_run(adapter, model, layer_idx, ids, position_ids=None):
    mgr = HookManager(adapter).capture_resid_pre([layer_idx])
    with mgr:
        with torch.no_grad():
            out = model(input_ids=ids, position_ids=position_ids)
    return mgr.captures[layer_idx], out


def test_recompute_qk_pre_rope_matches_hooked_qk_norm_outputs(tiny_qwen3_model):
    """Preview of check (c): recompute from a resid_pre captured via HookManager,
    stopping right after QK-norm, must match the real q_norm/k_norm outputs hooked
    during the same forward pass."""
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 1
    torch.manual_seed(2)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    ground_truth: dict[str, torch.Tensor] = {}
    h_q = adapter.q_norm(layer_idx).register_forward_hook(
        lambda m, a, out: ground_truth.__setitem__("q", out.detach().clone())
    )
    h_k = adapter.k_norm(layer_idx).register_forward_hook(
        lambda m, a, out: ground_truth.__setitem__("k", out.detach().clone())
    )
    try:
        resid_pre, _ = _capture_resid_pre_and_run(adapter, tiny_qwen3_model, layer_idx, ids)
    finally:
        h_q.remove()
        h_k.remove()

    q, k = recompute_qk_pre_rope(adapter, resid_pre, layer_idx)
    # q_norm/k_norm are applied before the transpose to [batch, heads, seq, head_dim]
    # in the real forward (hooked output is [batch, seq, heads, head_dim]); align shapes.
    torch.testing.assert_close(q, ground_truth["q"].transpose(1, 2))
    torch.testing.assert_close(k, ground_truth["k"].transpose(1, 2))


def test_recompute_qk_matches_real_forward_with_rotary(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 0
    torch.manual_seed(3)
    seq_len = 6
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling

    # apply_rotary_pos_emb is called once per layer per forward pass (in layer
    # order), so record every call and pick out layer_idx's — not a dict keyed by
    # a fixed name, which the next layer's call would silently overwrite.
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_fn = qwen3_modeling.apply_rotary_pos_emb

    def spy(q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = original_fn(q, k, cos, sin, unsqueeze_dim)
        calls.append((q_out.detach().clone(), k_out.detach().clone()))
        return q_out, k_out

    qwen3_modeling.apply_rotary_pos_emb = spy
    try:
        resid_pre, _ = _capture_resid_pre_and_run(adapter, tiny_qwen3_model, layer_idx, ids, position_ids)
    finally:
        qwen3_modeling.apply_rotary_pos_emb = original_fn

    q, k = recompute_qk(adapter, resid_pre, layer_idx, position_ids)
    expected_q, expected_k = calls[layer_idx]
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(k, expected_k)


def test_recompute_v_matches_real(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 1
    torch.manual_seed(4)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    ground_truth: dict[str, torch.Tensor] = {}
    h = adapter.v_proj(layer_idx).register_forward_hook(
        lambda m, a, out: ground_truth.__setitem__("v", out.detach().clone())
    )
    try:
        resid_pre, _ = _capture_resid_pre_and_run(adapter, tiny_qwen3_model, layer_idx, ids)
    finally:
        h.remove()

    v = recompute_v(adapter, resid_pre, layer_idx)
    expected = adapter.reshape_to_heads(ground_truth["v"], adapter.num_key_value_heads)
    torch.testing.assert_close(v, expected)


def test_recompute_attention_matches_real_eager_self_attn(tiny_qwen3_model):
    """Preview of check (d): force eager attention, hook self_attn's own
    (attn_output, attn_weights) return, and check recompute_attention reproduces it
    from independently-recomputed q/k/v. self_attn.forward's return is already
    post-o_proj and reassembled to [batch, seq, hidden], whereas recompute_attention
    stops right where eager_attention_forward does (pre-o_proj, per-head) — so
    attn_weights compares directly, and attn_output is pushed through the model's
    own o_proj before comparing."""
    tiny_qwen3_model.config._attn_implementation = "eager"
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 0
    torch.manual_seed(5)
    seq_len = 5
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    ground_truth: dict[str, torch.Tensor] = {}

    def self_attn_hook(module, args, output):
        attn_output, attn_weights = output
        ground_truth["attn_output"] = attn_output.detach().clone()
        ground_truth["attn_weights"] = attn_weights.detach().clone()

    self_attn = tiny_qwen3_model.model.layers[layer_idx].self_attn
    h = self_attn.register_forward_hook(self_attn_hook)
    try:
        resid_pre, _ = _capture_resid_pre_and_run(adapter, tiny_qwen3_model, layer_idx, ids, position_ids)
    finally:
        h.remove()

    q, k = recompute_qk(adapter, resid_pre, layer_idx, position_ids)
    v = recompute_v(adapter, resid_pre, layer_idx)
    attn_output, attn_weights = recompute_attention(adapter, q, k, v, layer_idx)
    torch.testing.assert_close(attn_weights, ground_truth["attn_weights"])

    batch, seq_len = attn_output.shape[0], attn_output.shape[1]
    reassembled = attn_output.reshape(batch, seq_len, -1)
    final_output = self_attn.o_proj(reassembled)
    torch.testing.assert_close(final_output, ground_truth["attn_output"])


def test_recompute_logits_matches_real_forward(tiny_qwen3_model):
    """Preview of check (a): logits rebuilt from the captured final residual
    (final norm -> lm_head) must match the model's own logits."""
    adapter = Qwen3Adapter(tiny_qwen3_model)
    torch.manual_seed(6)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    mgr = HookManager(adapter).capture_final_residual()
    with mgr:
        with torch.no_grad():
            out = tiny_qwen3_model(input_ids=ids)

    logits = recompute_logits(adapter, mgr.captures["final_residual"])
    torch.testing.assert_close(logits, out.logits)
