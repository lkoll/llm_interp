import pytest
import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.capture import CaptureRunner, trace_id_for
from llm_interp.generate import HFGenerator
from llm_interp.mlp import REGIME_NAMES, unpack_gate_sign_bits
from llm_interp.prompts import ArcEasyPromptSource, generate_pair

pytestmark = pytest.mark.gpu


def _short_generation_config(qwen3_run_config):
    return qwen3_run_config.generation.model_copy(update={"max_new_tokens": 48})


def _one_trace(qwen3_model_and_tokenizer, qwen3_run_config):
    model, tokenizer = qwen3_model_and_tokenizer
    generator = HFGenerator(model)
    item = ArcEasyPromptSource(n=1).load()[0]
    return generate_pair(item, generator, tokenizer, _short_generation_config(qwen3_run_config))


def test_mlp_dense_capture_matches_real_forward(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    traces = _one_trace(qwen3_model_and_tokenizer, qwen3_run_config)

    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [0],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [3], "topk_layers": [], "capture_validation_extras": False}
                    ),
                }
            )
        }
    )
    runner = CaptureRunner(config, model, tokenizer, adapter, tmp_path)
    runner.run(traces)

    trace_id = trace_id_for(traces[0].record.pair_id, traces[0].record.condition)
    row = next(r for r in runner.tables._rows["index"] if r["trace_id"] == trace_id)
    start, end = row["start"], row["end"]

    act = runner.zarr_store.read("mlp_act/3", start, end)
    gate_sign_packed = runner.zarr_store.read("mlp_gate_sign/3", start, end).numpy()
    d_mlp = adapter.intermediate_size
    gate_sign = unpack_gate_sign_bits(gate_sign_packed, d_mlp)

    # ground truth: hook gate_proj/down_proj directly on a fresh forward pass
    ground_truth = {}
    h_gate = adapter.mlp_gate_proj(3).register_forward_hook(
        lambda m, a, out: ground_truth.__setitem__("gate", out.detach().clone())
    )
    h_act = adapter.mlp_down_proj(3).register_forward_pre_hook(
        lambda m, args, kwargs: ground_truth.__setitem__("act", args[0].detach().clone()),
        with_kwargs=True,
    )
    try:
        full_ids = traces[0].generation_result.full_ids.unsqueeze(0).to(model.device)
        with torch.no_grad():
            model(input_ids=full_ids, use_cache=False)
    finally:
        h_gate.remove()
        h_act.remove()

    torch.testing.assert_close(act, ground_truth["act"][0].cpu(), atol=1e-4, rtol=1e-4)
    expected_gate_sign = (ground_truth["gate"][0].cpu() < 0).to(torch.uint8).numpy()
    import numpy as np

    np.testing.assert_array_equal(gate_sign, expected_gate_sign)


def test_mlp_topk_capture_round_trips_and_regimes_valid(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    traces = _one_trace(qwen3_model_and_tokenizer, qwen3_run_config)

    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [0],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [], "topk_layers": [5], "top_k": 8, "capture_validation_extras": False}
                    ),
                }
            )
        }
    )
    runner = CaptureRunner(config, model, tokenizer, adapter, tmp_path)
    runner.run(traces)

    trace_id = trace_id_for(traces[0].record.pair_id, traces[0].record.condition)
    row = next(r for r in runner.tables._rows["index"] if r["trace_id"] == trace_id)
    start, end = row["start"], row["end"]
    n_tokens = end - start

    indices = runner.zarr_store.read("mlp_topk_index/5", start, end)
    values = runner.zarr_store.read("mlp_topk_value/5", start, end)
    regime = runner.zarr_store.read("mlp_topk_regime/5", start, end)

    assert indices.shape == (n_tokens, 8)
    assert values.shape == (n_tokens, 8)
    assert regime.shape == (n_tokens, 8)
    assert indices.dtype == torch.int16
    assert regime.dtype == torch.uint8
    assert indices.min().item() >= 0
    assert indices.max().item() < adapter.intermediate_size
    for code in regime.flatten().tolist():
        assert code in REGIME_NAMES


def test_mlp_validation_extras_match_real_forward(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    traces = _one_trace(qwen3_model_and_tokenizer, qwen3_run_config)

    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [0],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [2], "topk_layers": [], "capture_validation_extras": True}
                    ),
                }
            )
        }
    )
    runner = CaptureRunner(config, model, tokenizer, adapter, tmp_path)
    runner.run(traces)

    trace_id = trace_id_for(traces[0].record.pair_id, traces[0].record.condition)
    row = next(r for r in runner.tables._rows["index"] if r["trace_id"] == trace_id)
    start, end = row["start"], row["end"]

    gate = runner.zarr_store.read("mlp_gate/2", start, end)
    up = runner.zarr_store.read("mlp_up/2", start, end)
    mlp_out = runner.zarr_store.read("mlp_out/2", start, end)
    act = runner.zarr_store.read("mlp_act/2", start, end)

    # act_fn(gate) * up == mlp_act (a preview of check g, 0.17)
    expected_act = adapter.mlp_activation_fn()(gate) * up
    torch.testing.assert_close(act, expected_act, atol=1e-4, rtol=1e-4)

    # down_proj(mlp_act) == mlp_out
    down_proj = adapter.mlp_down_proj(2)
    expected_out = down_proj(act.to(next(down_proj.parameters()).device)).cpu()
    torch.testing.assert_close(mlp_out, expected_out, atol=1e-4, rtol=1e-4)
