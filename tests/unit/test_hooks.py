import pytest
import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.hooks import HookManager


def _num_pre_hooks(module) -> int:
    return len(module._forward_pre_hooks)


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
