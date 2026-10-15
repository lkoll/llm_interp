import pytest
import torch

from llm_interp.loader import Loader
from llm_interp.store import TraceTables, ZarrStore


class _FakeTokenizer:
    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.vocab[i] for i in ids]


def _build_fake_run(root, hidden_size=4):
    """Two traces: 't1' (5 tokens), 't2' (3 tokens), each with a resid_pre/0 site
    and token_ids, built directly (no model) to test the loader in isolation."""
    store = ZarrStore(root / "activations.zarr")
    tables = TraceTables(root)

    t1_ids = torch.tensor([10, 11, 12, 13, 14])
    t1_resid = torch.arange(5 * hidden_size, dtype=torch.float32).reshape(5, hidden_size)
    s, e = store.append("positions", torch.arange(5))
    store.append("token_ids", t1_ids)
    store.append("resid_pre/0", t1_resid)
    tables.add_index_row({"trace_id": "t1", "start": s, "end": e})
    tables.add_traces_row(
        {
            "trace_id": "t1",
            "pair_id": "p1",
            "condition": "no_hint",
            "hinted_option": None,
            "gold": "A",
            "parsed_answer": "A",
            "parse_status": "ok",
            "stop_reason": "eos",
            "truncated": False,
            "seed": 0,
            "prompt_len": 2,
            "total_len": 5,
            "has_think_close": True,
        }
    )
    tables.add_spans_rows([{"trace_id": "t1", "span_name": "think", "start": 1, "end": 3}])

    t2_ids = torch.tensor([20, 21, 22])
    t2_resid = torch.arange(3 * hidden_size, dtype=torch.float32).reshape(3, hidden_size) + 1000
    s2, e2 = store.append("positions", torch.arange(3))
    store.append("token_ids", t2_ids)
    store.append("resid_pre/0", t2_resid)
    tables.add_index_row({"trace_id": "t2", "start": s2, "end": e2})
    tables.add_traces_row(
        {
            "trace_id": "t2",
            "pair_id": "p1",
            "condition": "hint",
            "hinted_option": "B",
            "gold": "A",
            "parsed_answer": "B",
            "parse_status": "ok",
            "stop_reason": "max_new_tokens",
            "truncated": True,
            "seed": 1,
            "prompt_len": 2,
            "total_len": 3,
            "has_think_close": False,
        }
    )
    tables.flush()
    return {"t1_ids": t1_ids, "t1_resid": t1_resid, "t2_ids": t2_ids, "t2_resid": t2_resid}


@pytest.fixture
def vocab():
    return {i: f"<tok{i}>" for i in range(10, 25)}


def test_loader_trace_ids(tmp_path, vocab):
    _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    assert set(loader.trace_ids()) == {"t1", "t2"}


def test_loader_trace_metadata(tmp_path, vocab):
    _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    meta = loader.trace_metadata("t2")
    assert meta["condition"] == "hint"
    assert meta["hinted_option"] == "B"


def test_loader_spans_for(tmp_path, vocab):
    _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    assert loader.spans_for("t1") == [{"trace_id": "t1", "span_name": "think", "start": 1, "end": 3}]
    assert loader.spans_for("t2") == []


def test_loader_load_full_trace(tmp_path, vocab):
    data = _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    loaded = loader.load("t1", "resid_pre", layer=0)
    torch.testing.assert_close(loaded.tensor, data["t1_resid"])
    assert loaded.token_ids == data["t1_ids"].tolist()
    assert loaded.decoded_tokens == [vocab[i] for i in data["t1_ids"].tolist()]


def test_loader_load_second_trace_uses_correct_offset(tmp_path, vocab):
    """t2 lives at a nonzero offset in the global zarr arrays; the loader must
    translate the trace-relative span into the right absolute slice."""
    data = _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    loaded = loader.load("t2", "resid_pre", layer=0)
    torch.testing.assert_close(loaded.tensor, data["t2_resid"])
    assert loaded.token_ids == data["t2_ids"].tolist()


def test_loader_load_partial_span(tmp_path, vocab):
    data = _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    loaded = loader.load("t1", "resid_pre", layer=0, span=(1, 3))
    torch.testing.assert_close(loaded.tensor, data["t1_resid"][1:3])
    assert loaded.token_ids == data["t1_ids"].tolist()[1:3]


def test_loader_load_site_without_layer(tmp_path, vocab):
    data = _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    loaded = loader.load("t1", "token_ids")
    assert loaded.token_ids == data["t1_ids"].tolist()


def test_loader_unknown_trace_id_raises(tmp_path, vocab):
    _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    with pytest.raises(KeyError):
        loader.load("nonexistent", "resid_pre", layer=0)


def test_loader_span_out_of_bounds_raises(tmp_path, vocab):
    _build_fake_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    with pytest.raises(ValueError):
        loader.load("t1", "resid_pre", layer=0, span=(0, 100))
