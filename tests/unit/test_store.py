import torch

from llm_interp.store import TraceTables, ZarrStore
from llm_interp.store.manifest import build_manifest, write_manifest


class _FakeTokenizer:
    chat_template = "{{ messages }}"

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1}


def test_zarr_store_round_trip_fp32_bit_exact(tmp_path):
    store = ZarrStore(tmp_path / "activations.zarr")
    data = torch.randn(5, 4, dtype=torch.float32)
    start, end = store.append("resid_pre/0", data)
    assert (start, end) == (0, 5)
    torch.testing.assert_close(store.read("resid_pre/0", start, end), data, atol=0, rtol=0)


def test_zarr_store_round_trip_bf16_bit_exact(tmp_path):
    """Check (e): bit-exact storage round trip, bf16 path (stored as raw uint16)."""
    store = ZarrStore(tmp_path / "activations.zarr")
    data = torch.randn(5, 4, dtype=torch.float32).to(torch.bfloat16)
    start, end = store.append("resid_pre/0", data)
    read_back = store.read("resid_pre/0", start, end)
    assert read_back.dtype == torch.bfloat16
    assert torch.equal(read_back, data)


def test_zarr_store_round_trip_fp32_bit_exact_strict_equal(tmp_path):
    """Check (e): bit-exact storage round trip, fp32 path."""
    store = ZarrStore(tmp_path / "activations.zarr")
    data = torch.randn(5, 4, dtype=torch.float32)
    start, end = store.append("resid_pre/0", data)
    assert torch.equal(store.read("resid_pre/0", start, end), data)


def test_zarr_store_append_is_sequential_along_token_axis(tmp_path):
    store = ZarrStore(tmp_path / "activations.zarr")
    first = torch.randn(3, 4)
    second = torch.randn(2, 4)
    s1, e1 = store.append("site", first)
    s2, e2 = store.append("site", second)
    assert (s1, e1) == (0, 3)
    assert (s2, e2) == (3, 5)
    torch.testing.assert_close(store.read("site", 0, 5), torch.cat([first, second], dim=0))


def test_zarr_store_truncate(tmp_path):
    store = ZarrStore(tmp_path / "activations.zarr")
    store.append("positions", torch.arange(10))
    store.truncate("positions", 4)
    assert store.group["positions"].shape[0] == 4


def test_zarr_store_truncate_all_handles_nested_groups(tmp_path):
    store = ZarrStore(tmp_path / "activations.zarr")
    store.append("resid_pre/0", torch.randn(10, 4))
    store.append("resid_pre/1", torch.randn(10, 4))
    store.append("final_residual", torch.randn(10, 4))

    store.truncate_all(6)

    assert store.group["resid_pre/0"].shape[0] == 6
    assert store.group["resid_pre/1"].shape[0] == 6
    assert store.group["final_residual"].shape[0] == 6


def test_zarr_store_truncate_all_is_noop_on_shorter_arrays(tmp_path):
    store = ZarrStore(tmp_path / "activations.zarr")
    store.append("site", torch.randn(3, 4))
    store.truncate_all(10)  # longer than current length -> no-op, not an error
    assert store.group["site"].shape[0] == 3


def test_zarr_store_reopens_persisted_data(tmp_path):
    path = tmp_path / "activations.zarr"
    ZarrStore(path).append("site", torch.randn(3, 4))
    reopened = ZarrStore(path)
    assert reopened.group["site"].shape[0] == 3


def _index_row(trace_id: str, start: int, end: int) -> dict:
    return {"trace_id": trace_id, "start": start, "end": end}


def _traces_row(trace_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "pair_id": "p1",
        "condition": "no_hint",
        "hinted_option": None,
        "gold": "A",
        "parsed_answer": "A",
        "parse_status": "ok",
        "stop_reason": "eos",
        "truncated": False,
        "seed": 0,
        "prompt_len": 10,
        "total_len": 20,
        "has_think_close": True,
    }


def test_trace_tables_round_trip(tmp_path):
    tables = TraceTables(tmp_path)
    tables.add_index_row(_index_row("t1", 0, 20))
    tables.add_traces_row(_traces_row("t1"))
    tables.add_spans_rows([{"trace_id": "t1", "span_name": "think", "start": 1, "end": 5}])
    tables.flush()

    reloaded = TraceTables(tmp_path)
    assert reloaded.done_trace_ids() == {"t1"}
    assert reloaded.last_index_end() == 20
    assert len(reloaded._rows["traces"]) == 1
    assert len(reloaded._rows["spans"]) == 1


def test_trace_tables_last_index_end_is_zero_when_empty(tmp_path):
    tables = TraceTables(tmp_path)
    assert tables.last_index_end() == 0
    assert tables.done_trace_ids() == set()


def test_trace_tables_drops_orphaned_rows_not_in_index(tmp_path):
    """Simulates a crash between flushing traces/spans and flushing index: on
    reload, rows for a trace_id absent from `index` must be dropped."""
    tables = TraceTables(tmp_path)
    tables.add_index_row(_index_row("t1", 0, 20))
    tables.add_traces_row(_traces_row("t1"))
    tables.add_traces_row(_traces_row("t2"))  # orphan: never made it into index
    tables.add_spans_rows([{"trace_id": "t2", "span_name": "think", "start": 1, "end": 5}])
    tables.flush()

    reloaded = TraceTables(tmp_path)
    assert len(reloaded._rows["traces"]) == 1
    assert reloaded._rows["traces"][0]["trace_id"] == "t1"
    assert reloaded._rows["spans"] == []


def test_build_manifest_has_expected_keys():
    from llm_interp.config import RunConfig

    config = RunConfig.model_validate(
        {
            "run_name": "unit_test",
            "model": {
                "id": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "dtype": "float32",
                "device": "cpu",
                "attn_implementation": "sdpa",
            },
            "capture": {"layers": [0]},
            "generation": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": 8,
                "seed": 0,
            },
            "storage_root": "runs/unit_test",
        }
    )
    manifest = build_manifest(config, _FakeTokenizer())
    for key in (
        "created_at",
        "git_commit",
        "package_versions",
        "cuda_version",
        "gpu",
        "model_id",
        "model_revision",
        "tokenizer_hash",
        "chat_template_hash",
        "config",
    ):
        assert key in manifest
    assert manifest["package_versions"]["torch"] is not None
    assert manifest["config"]["run_name"] == "unit_test"


def test_write_manifest_round_trips(tmp_path):
    manifest = {"a": 1, "nested": {"b": 2}}
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    import json

    assert json.loads(path.read_text()) == manifest
