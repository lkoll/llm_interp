import pytest
import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.hooks import HookManager
from llm_interp.recompute import recompute_logits, recompute_qk_pre_rope

pytestmark = pytest.mark.gpu


def test_check_a_logits_from_final_residual(qwen3_model_and_tokenizer):
    """Check (a): logits rebuilt from the final residual (final norm -> lm_head)
    match the model's own logits, on the real Qwen3-0.6B."""
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)

    ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.to(model.device)

    mgr = HookManager(adapter).capture_final_residual()
    with mgr:
        with torch.no_grad():
            out = model(input_ids=ids)

    recomputed = recompute_logits(adapter, mgr.captures["final_residual"])
    torch.testing.assert_close(recomputed, out.logits, atol=1e-4, rtol=1e-4)


def test_check_c_recomputed_qk_matches_hooked_qk_norm(qwen3_model_and_tokenizer):
    """Check (c): q/k recomputed from resid_pre (input norm -> proj -> reshape ->
    QK-norm) match the hooked q_norm/k_norm outputs, across every layer of the real
    Qwen3-0.6B."""
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    all_layers = list(range(adapter.num_layers))

    ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.to(model.device)

    ground_truth: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(layer_idx, slot):
        def hook(module, args, output):
            q, k = ground_truth.get(layer_idx, (None, None))
            if slot == "q":
                ground_truth[layer_idx] = (output.detach().clone(), k)
            else:
                ground_truth[layer_idx] = (q, output.detach().clone())

        return hook

    handles = []
    for layer_idx in all_layers:
        handles.append(adapter.q_norm(layer_idx).register_forward_hook(make_hook(layer_idx, "q")))
        handles.append(adapter.k_norm(layer_idx).register_forward_hook(make_hook(layer_idx, "k")))

    mgr = HookManager(adapter).capture_resid_pre(all_layers)
    try:
        with mgr:
            with torch.no_grad():
                model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    for layer_idx in all_layers:
        resid_pre = mgr.captures[layer_idx]
        q, k = recompute_qk_pre_rope(adapter, resid_pre, layer_idx)
        expected_q, expected_k = ground_truth[layer_idx]
        # hooked q_norm/k_norm outputs are [batch, seq, heads, head_dim], pre-transpose
        torch.testing.assert_close(q, expected_q.transpose(1, 2), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k, expected_k.transpose(1, 2), atol=1e-4, rtol=1e-4)
