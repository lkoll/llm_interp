import pytest
import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.hooks import HookManager


def _num_pre_hooks(module) -> int:
    return len(module._forward_pre_hooks)


def _num_forward_hooks(module) -> int:
    return len(module._forward_hooks)


def test_capture_resid_pre_layer0_matches_embeddings(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    with torch.no_grad():
        expected = adapter.embeddings()(ids).detach().clone()

    mgr = HookManager(adapter).capture_resid_pre([0])
    with mgr:
        with torch.no_grad():
            tiny_qwen3_model(input_ids=ids)

    torch.testing.assert_close(mgr.captures[0], expected)


def test_capture_resid_pre_layer1_matches_layer0_output(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    ids = torch.randint(1, tiny_qwen3_model.config.vocab_size, (1, 5))

    ground_truth: dict[str, torch.Tensor] = {}
    h = adapter.layer(0).register_forward_hook(
        lambda module, args, output: ground_truth.__setitem__("layer0_out", output.detach().clone())
    )
    mgr = HookManager(adapter).capture_resid_pre([1])
    try:
        with mgr:
            with torch.no_grad():
                tiny_qwen3_model(input_ids=ids)
    finally:
        h.remove()

    torch.testing.assert_close(mgr.captures[1], ground_truth["layer0_out"])


def test_captures_are_detached_cloned_cpu(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    original_ref: dict[str, torch.Tensor] = {}
    h = adapter.layer(0).register_forward_pre_hook(
        lambda module, args, kwargs: original_ref.__setitem__("hidden_states", args[0]),
        with_kwargs=True,
    )
    mgr = HookManager(adapter).capture_resid_pre([0])
    try:
        with mgr:
            with torch.no_grad():
                tiny_qwen3_model(input_ids=ids)
    finally:
        h.remove()

    captured = mgr.captures[0]
    assert captured.requires_grad is False
    assert captured.device.type == "cpu"
    assert captured.data_ptr() != original_ref["hidden_states"].data_ptr()


def test_hooks_removed_on_normal_exit(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer0 = adapter.layer(0)
    assert _num_pre_hooks(layer0) == 0

    mgr = HookManager(adapter).capture_resid_pre([0])
    with mgr:
        assert _num_pre_hooks(layer0) == 1

    assert _num_pre_hooks(layer0) == 0


def test_hooks_removed_on_exception(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer0 = adapter.layer(0)

    mgr = HookManager(adapter).capture_resid_pre([0])
    with pytest.raises(RuntimeError):
        with mgr:
            assert _num_pre_hooks(layer0) == 1
            raise RuntimeError("boom")

    assert _num_pre_hooks(layer0) == 0


def test_add_zero_vector_leaves_logits_unchanged(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    with torch.no_grad():
        baseline = tiny_qwen3_model(input_ids=ids).logits

    zero_vector = torch.zeros(adapter.hidden_size)
    mgr = HookManager(adapter).add_vector(1, zero_vector)
    with mgr:
        with torch.no_grad():
            with_zero = tiny_qwen3_model(input_ids=ids).logits

    torch.testing.assert_close(with_zero, baseline)


def test_add_nonzero_vector_changes_logits(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    with torch.no_grad():
        baseline = tiny_qwen3_model(input_ids=ids).logits

    nonzero_vector = torch.ones(adapter.hidden_size) * 10.0
    mgr = HookManager(adapter).add_vector(1, nonzero_vector)
    with mgr:
        with torch.no_grad():
            with_vector = tiny_qwen3_model(input_ids=ids).logits

    assert not torch.allclose(with_vector, baseline)


def test_capture_mlp_sites_match_real_forward(tiny_qwen3_model):
    """Addendum A: mlp_in/mlp_gate/mlp_up/mlp_act/mlp_out, all captured in one
    pass, each checked against an independent hook on the real model."""
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 1
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    ground_truth: dict[str, torch.Tensor] = {}
    handles = [
        adapter.mlp(layer_idx).register_forward_pre_hook(
            lambda m, args, kwargs: ground_truth.__setitem__("mlp_in", args[0].detach().clone()),
            with_kwargs=True,
        ),
        adapter.mlp(layer_idx).register_forward_hook(
            lambda m, args, output: ground_truth.__setitem__("mlp_out", output.detach().clone())
        ),
        adapter.mlp_gate_proj(layer_idx).register_forward_hook(
            lambda m, args, output: ground_truth.__setitem__("mlp_gate", output.detach().clone())
        ),
        adapter.mlp_up_proj(layer_idx).register_forward_hook(
            lambda m, args, output: ground_truth.__setitem__("mlp_up", output.detach().clone())
        ),
        adapter.mlp_down_proj(layer_idx).register_forward_pre_hook(
            lambda m, args, kwargs: ground_truth.__setitem__("mlp_act", args[0].detach().clone()),
            with_kwargs=True,
        ),
    ]

    mgr = (
        HookManager(adapter)
        .capture_mlp_in([layer_idx])
        .capture_mlp_gate([layer_idx])
        .capture_mlp_up([layer_idx])
        .capture_mlp_act([layer_idx])
        .capture_mlp_out([layer_idx])
    )
    try:
        with mgr:
            with torch.no_grad():
                tiny_qwen3_model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    torch.testing.assert_close(mgr.captures[("mlp_in", layer_idx)], ground_truth["mlp_in"])
    torch.testing.assert_close(mgr.captures[("mlp_gate", layer_idx)], ground_truth["mlp_gate"])
    torch.testing.assert_close(mgr.captures[("mlp_up", layer_idx)], ground_truth["mlp_up"])
    torch.testing.assert_close(mgr.captures[("mlp_act", layer_idx)], ground_truth["mlp_act"])
    torch.testing.assert_close(mgr.captures[("mlp_out", layer_idx)], ground_truth["mlp_out"])


def test_mlp_act_equals_act_fn_gate_times_up(tiny_qwen3_model):
    """mlp_act is act_fn(gate) * up by definition (Addendum A) — check the
    captured values satisfy that relationship, not just that they were captured."""
    adapter = Qwen3Adapter(tiny_qwen3_model)
    layer_idx = 0
    ids = torch.randint(0, tiny_qwen3_model.config.vocab_size, (1, 5))

    mgr = HookManager(adapter).capture_mlp_gate([layer_idx]).capture_mlp_up([layer_idx]).capture_mlp_act([layer_idx])
    with mgr:
        with torch.no_grad():
            tiny_qwen3_model(input_ids=ids)

    gate = mgr.captures[("mlp_gate", layer_idx)]
    up = mgr.captures[("mlp_up", layer_idx)]
    act = mgr.captures[("mlp_act", layer_idx)]
    expected = adapter.mlp_activation_fn()(gate) * up
    torch.testing.assert_close(act, expected)


def test_output_capture_hooks_removed_on_exit(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    gate_proj = adapter.mlp_gate_proj(0)
    assert _num_forward_hooks(gate_proj) == 0

    mgr = HookManager(adapter).capture_mlp_gate([0])
    with mgr:
        assert _num_forward_hooks(gate_proj) == 1

    assert _num_forward_hooks(gate_proj) == 0


def test_output_capture_hooks_removed_on_exception(tiny_qwen3_model):
    adapter = Qwen3Adapter(tiny_qwen3_model)
    gate_proj = adapter.mlp_gate_proj(0)

    mgr = HookManager(adapter).capture_mlp_gate([0])
    with pytest.raises(RuntimeError):
        with mgr:
            assert _num_forward_hooks(gate_proj) == 1
            raise RuntimeError("boom")

    assert _num_forward_hooks(gate_proj) == 0
