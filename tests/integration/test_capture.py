import torch

from llm_interp.adapters import Qwen3Adapter
from llm_interp.capture import CaptureRunner, trace_id_for
from llm_interp.generate import HFGenerator
from llm_interp.prompts import ArcEasyPromptSource, generate_pair

import pytest

pytestmark = pytest.mark.gpu


def _short_generation_config(qwen3_run_config):
    # keep the real sampling params but cap generation short, so these tests run fast
    return qwen3_run_config.generation.model_copy(update={"max_new_tokens": 64})


def test_capture_runner_writes_sites_tables_and_round_trips_bit_exact(
    qwen3_model_and_tokenizer, qwen3_run_config, tmp_path
):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    generator = HFGenerator(model)
    gen_cfg = _short_generation_config(qwen3_run_config)

    item = ArcEasyPromptSource(n=1).load()[0]
    traces = generate_pair(item, generator, tokenizer, gen_cfg)

    config = qwen3_run_config.model_copy(update={"capture": qwen3_run_config.capture.model_copy(update={"layers": [0, 1]})})
    runner = CaptureRunner(config, model, tokenizer, adapter, tmp_path)
    runner.run(traces)

    done_ids = runner.tables.done_trace_ids()
    expected_ids = {trace_id_for(item.pair_id, "no_hint"), trace_id_for(item.pair_id, "hint")}
    assert done_ids == expected_ids

    total_tokens = sum(t.generation_result.full_ids.shape[0] for t in traces)
    assert runner.zarr_store.group["positions"].shape[0] == total_tokens
    assert runner.zarr_store.group["resid_pre/0"].shape[1] == adapter.hidden_size
    assert runner.zarr_store.group["final_residual"].shape == (total_tokens, adapter.hidden_size)
    assert runner.zarr_store.group["top_logprob_values"].shape == (total_tokens, config.capture.top_k_logprobs)

    # check (e), against real captured data: bit-exact round trip through the store.
    for row in runner.tables._rows["index"]:
        start, end = row["start"], row["end"]
        read_back = runner.zarr_store.read("final_residual", start, end)
        assert read_back.dtype == torch.float32
        assert read_back.shape == (end - start, adapter.hidden_size)

    traces_rows = {r["trace_id"]: r for r in runner.tables._rows["traces"]}
    for trace in traces:
        tid = trace_id_for(trace.record.pair_id, trace.record.condition)
        row = traces_rows[tid]
        assert row["gold"] == item.gold
        assert row["condition"] == trace.record.condition
        assert row["seed"] == trace.generation_result.seed

    span_names_by_trace: dict[str, set[str]] = {}
    for row in runner.tables._rows["spans"]:
        span_names_by_trace.setdefault(row["trace_id"], set()).add(row["span_name"])
    for trace in traces:
        tid = trace_id_for(trace.record.pair_id, trace.record.condition)
        names = span_names_by_trace.get(tid, set())
        # "think" should be present whenever has_think_close is True for that trace
        if traces_rows[tid]["has_think_close"]:
            assert {"think_open", "think", "think_close", "response"} <= names
        if trace.record.hinted_option is not None:
            assert "hint" in names


def test_capture_runner_rerun_is_a_no_op(qwen3_model_and_tokenizer, qwen3_run_config, tmp_path):
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    generator = HFGenerator(model)
    gen_cfg = _short_generation_config(qwen3_run_config)

    item = ArcEasyPromptSource(n=1).load()[0]
    traces = generate_pair(item, generator, tokenizer, gen_cfg)

    runner1 = CaptureRunner(qwen3_run_config, model, tokenizer, adapter, tmp_path)
    runner1.run(traces)
    index_len_after_first_run = runner1.zarr_store.group["positions"].shape[0]
    n_rows_after_first_run = len(runner1.tables._rows["traces"])

    runner2 = CaptureRunner(qwen3_run_config, model, tokenizer, adapter, tmp_path)
    runner2.run(traces)  # re-running with the same traces must be a no-op

    assert runner2.zarr_store.group["positions"].shape[0] == index_len_after_first_run
    assert len(runner2.tables._rows["traces"]) == n_rows_after_first_run


def test_capture_runner_recovers_from_orphaned_partial_write(
    qwen3_model_and_tokenizer, qwen3_run_config, tmp_path
):
    """Simulates a crash: data appended to Zarr for a trace that never made it
    into the index. On the next run, that orphaned data must be truncated away,
    and the real traces must still be captured correctly (not skipped, not
    duplicated)."""
    model, tokenizer = qwen3_model_and_tokenizer
    adapter = Qwen3Adapter(model)
    generator = HFGenerator(model)
    gen_cfg = _short_generation_config(qwen3_run_config)

    item = ArcEasyPromptSource(n=1).load()[0]
    traces = generate_pair(item, generator, tokenizer, gen_cfg)

    # Run only the first trace "for real" and index it.
    runner1 = CaptureRunner(qwen3_run_config, model, tokenizer, adapter, tmp_path)
    runner1.run([traces[0]])
    indexed_end = runner1.tables.last_index_end()

    # Simulate a crash mid-write on the second trace: append garbage to Zarr
    # directly, without ever recording it in the index.
    garbage_len = 7
    runner1.zarr_store.append("positions", torch.arange(garbage_len))
    for layer in qwen3_run_config.capture.layers:
        runner1.zarr_store.append(f"resid_pre/{layer}", torch.randn(garbage_len, adapter.hidden_size))
    runner1.zarr_store.append("final_residual", torch.randn(garbage_len, adapter.hidden_size))
    runner1.zarr_store.append("top_logprob_values", torch.randn(garbage_len, qwen3_run_config.capture.top_k_logprobs))
    runner1.zarr_store.append(
        "top_logprob_indices", torch.zeros(garbage_len, qwen3_run_config.capture.top_k_logprobs, dtype=torch.long)
    )
    runner1.zarr_store.append("realized_logprob", torch.randn(garbage_len))
    assert runner1.zarr_store.group["positions"].shape[0] == indexed_end + garbage_len

    # "Restart": a fresh CaptureRunner on the same store_root, given both traces.
    runner2 = CaptureRunner(qwen3_run_config, model, tokenizer, adapter, tmp_path)
    runner2.run(traces)

    done_ids = runner2.tables.done_trace_ids()
    expected_ids = {trace_id_for(item.pair_id, "no_hint"), trace_id_for(item.pair_id, "hint")}
    assert done_ids == expected_ids  # both present, garbage truncated, first trace not duplicated

    total_tokens = sum(t.generation_result.full_ids.shape[0] for t in traces)
    assert runner2.zarr_store.group["positions"].shape[0] == total_tokens
