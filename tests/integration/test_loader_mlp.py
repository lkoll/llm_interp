import pytest
import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.capture import CaptureRunner, trace_id_for
from llm_interp.generate import HFGenerator
from llm_interp.loader import Loader
from llm_interp.mlp import unpack_gate_sign_bits
from llm_interp.prompts import ArcEasyPromptSource, generate_pair

pytestmark = pytest.mark.gpu


def test_get_neuron_dense_matches_real_capture(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    generator = HFGenerator(model)
    gen_cfg = qwen3_run_config.generation.model_copy(update={"max_new_tokens": 32})
    item = ArcEasyPromptSource(n=1).load()[0]
    traces = generate_pair(item, generator, tokenizer, gen_cfg)

    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [],
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

    loader = Loader(tmp_path, tokenizer)
    trace_id = trace_id_for(traces[0].record.pair_id, traces[0].record.condition)

    idx = 7
    result = loader.get_neuron(trace_id, layer=3, idx=idx)
    assert torch.all(result.available)

    direct_act = loader.load(trace_id, "mlp_act", layer=3).tensor
    torch.testing.assert_close(result.values, direct_act[:, idx])


def test_get_neuron_topk_matches_real_capture(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    generator = HFGenerator(model)
    gen_cfg = qwen3_run_config.generation.model_copy(update={"max_new_tokens": 32})
    item = ArcEasyPromptSource(n=1).load()[0]
    traces = generate_pair(item, generator, tokenizer, gen_cfg)

    config = qwen3_run_config.model_copy(
        update={
            "capture": qwen3_run_config.capture.model_copy(
                update={
                    "layers": [],
                    "capture_final_residual": False,
                    "mlp": qwen3_run_config.capture.mlp.model_copy(
                        update={"dense_layers": [], "topk_layers": [5], "top_k": 4, "capture_validation_extras": False}
                    ),
                }
            )
        }
    )
    runner = CaptureRunner(config, model, tokenizer, adapter, tmp_path)
    runner.run(traces)

    loader = Loader(tmp_path, tokenizer)
    trace_id = trace_id_for(traces[0].record.pair_id, traces[0].record.condition)

    indices = loader.load(trace_id, "mlp_topk_index", layer=5).tensor
    values = loader.load(trace_id, "mlp_topk_value", layer=5).tensor

    # pick a neuron index that was actually kept for at least one token
    kept_idx = indices[0, 0].item()
    result = loader.get_neuron(trace_id, layer=5, idx=kept_idx)
    assert result.available[0].item() is True
    assert result.values[0].item() == pytest.approx(values[0, 0].item())

    # a neuron index guaranteed not kept for token 0 (anything not in its row)
    not_kept = next(i for i in range(adapter.intermediate_size) if i not in indices[0].tolist())
    result_missing = loader.get_neuron(trace_id, layer=5, idx=not_kept)
    assert result_missing.available[0].item() is False
