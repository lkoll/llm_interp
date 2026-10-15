import torch

from .adapters.base import ArchAdapter


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def recompute_qk_pre_rope(
    adapter: ArchAdapter, resid_pre: torch.Tensor, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """input norm -> q/k proj -> per-head reshape -> QK-norm. No RoPE yet — this is
    the intermediate checked against the hooked q/k norm outputs (check c).

    `resid_pre` is moved to the model's device first: captures live on CPU
    (CLAUDE.md), but the real modules used for recompute may be on GPU.
    """
    norm = adapter.attn_input_norm(layer_idx)
    resid_pre = resid_pre.to(_module_device(norm))
    normed = norm(resid_pre)
    q = adapter.reshape_to_heads(adapter.q_proj(layer_idx)(normed), adapter.num_attention_heads)
    k = adapter.reshape_to_heads(adapter.k_proj(layer_idx)(normed), adapter.num_key_value_heads)
    return adapter.apply_qk_norm(q, k, layer_idx)


def recompute_qk(
    adapter: ArchAdapter, resid_pre: torch.Tensor, layer_idx: int, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full q/k recompute: input norm -> q/k proj -> reshape -> QK-norm -> RoPE at
    the given positions, using the model's own rotary module."""
    q, k = recompute_qk_pre_rope(adapter, resid_pre, layer_idx)
    position_ids = position_ids.to(q.device)
    return adapter.apply_rotary(q, k, position_ids, layer_idx)


def recompute_v(adapter: ArchAdapter, resid_pre: torch.Tensor, layer_idx: int) -> torch.Tensor:
    norm = adapter.attn_input_norm(layer_idx)
    resid_pre = resid_pre.to(_module_device(norm))
    normed = norm(resid_pre)
    return adapter.reshape_to_heads(adapter.v_proj(layer_idx)(normed), adapter.num_key_value_heads)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[batch, num_kv_heads, seq, head_dim] -> [batch, num_kv_heads * n_rep, seq, head_dim].

    Query head h reads KV head h // n_rep (GQA mapping); repeating each KV head
    n_rep times contiguously reproduces exactly that mapping.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _causal_additive_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.triu(mask, diagonal=1)


def recompute_attention(
    adapter: ArchAdapter, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA repeat + the model's own scaling + a causal mask (unpadded, batch size 1).

    Returns (attn_output, attn_weights) with attn_output shaped
    [batch, seq, num_heads, head_dim] — pre-o_proj, pre-reassembly — matching the
    model's own eager attention convention so it's directly comparable (check d).
    """
    n_rep = adapter.num_attention_heads // adapter.num_key_value_heads
    k_rep = repeat_kv(k, n_rep)
    v_rep = repeat_kv(v, n_rep)

    scaling = adapter.attention_scaling(layer_idx)
    seq_len = q.shape[-2]
    causal_mask = _causal_additive_mask(seq_len, dtype=q.dtype, device=q.device)

    attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * scaling + causal_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_rep)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def recompute_logits(adapter: ArchAdapter, final_residual: torch.Tensor) -> torch.Tensor:
    """final norm -> lm_head, for check (a). `final_residual` is moved to the
    model's device first; captures live on CPU."""
    norm = adapter.final_norm()
    final_residual = final_residual.to(_module_device(norm))
    normed = norm(final_residual)
    return adapter.lm_head()(normed)
