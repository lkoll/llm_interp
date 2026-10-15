import pytest
import torch

from llm_interp.loader import Loader
from llm_interp.mlp import pack_gate_sign_bits, regime_from_signs
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


def _build_fake_mlp_run(root, d_mlp=4):
    """One trace 't1', 4 tokens, with a dense MLP layer (0) and a topk MLP
    layer (1, k=2) — built directly to test get_neuron against both storage
    modes without needing a real model."""
    store = ZarrStore(root / "activations.zarr")
    tables = TraceTables(root)

    ids = torch.tensor([10, 11, 12, 13])
    s, e = store.append("positions", torch.arange(4))
    store.append("token_ids", ids)

    # Dense layer 0: handcrafted gate/act so get_neuron's output is predictable.
    gate = torch.tensor([[1.0, -1.0, 2.0, -2.0], [1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0], [2.0, -2.0, 2.0, -2.0]])
    act = torch.tensor([[3.0, -3.0, 0.1, -0.1], [1.0, 2.0, -0.2, -0.3], [-0.4, -0.5, 1.0, 2.0], [4.0, -4.0, 5.0, -5.0]])
    store.append("mlp_act/0", act)
    store.append("mlp_gate_sign/0", torch.from_numpy(pack_gate_sign_bits(gate)))

    # Topk layer 1, k=2: only the two largest |a| entries per token are kept.
    from llm_interp.mlp import topk_by_abs

    gate1 = torch.randn(4, d_mlp)
    act1 = torch.tensor([[1.0, 5.0, 2.0, 0.5], [3.0, 1.0, 4.0, 0.2], [0.1, 0.2, 0.3, 0.4], [9.0, 1.0, 2.0, 3.0]])
    indices, values, regime = topk_by_abs(act1, gate1, k=2)
    store.append("mlp_topk_index/1", indices)
    store.append("mlp_topk_value/1", values)
    store.append("mlp_topk_regime/1", regime)

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
            "total_len": 4,
            "has_think_close": True,
        }
    )
    tables.flush()
    return {"gate": gate, "act": act, "act1": act1, "indices": indices, "values": values, "regime": regime}


def test_get_neuron_dense_matches_handcrafted_values(tmp_path, vocab):
    data = _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    result = loader.get_neuron("t1", layer=0, idx=1)
    torch.testing.assert_close(result.values, data["act"][:, 1])
    assert torch.all(result.available)
    expected_regime = regime_from_signs(data["gate"][:, 1], data["act"][:, 1])
    assert torch.equal(result.regime, expected_regime)


def test_get_neuron_dense_respects_span(tmp_path, vocab):
    data = _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    result = loader.get_neuron("t1", layer=0, idx=2, span=(1, 3))
    torch.testing.assert_close(result.values, data["act"][1:3, 2])


def test_get_neuron_topk_available_where_kept(tmp_path, vocab):
    data = _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    # token 0's top-2 by |act1| are indices 1 (5.0) and 2 (2.0)
    result = loader.get_neuron("t1", layer=1, idx=1)
    assert result.available[0].item() is True
    assert result.values[0].item() == pytest.approx(5.0)


def test_get_neuron_topk_unavailable_where_dropped(tmp_path, vocab):
    _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    # token 0's act1 row is [1.0, 5.0, 2.0, 0.5] -> top-2 kept are idx 1, 2; idx 3 (0.5) is dropped
    result = loader.get_neuron("t1", layer=1, idx=3)
    assert result.available[0].item() is False
    assert torch.isnan(result.values[0])
    assert result.regime[0].item() == 255


def test_get_neuron_raises_for_layer_with_no_mlp_capture(tmp_path, vocab):
    _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))
    with pytest.raises(KeyError):
        loader.get_neuron("t1", layer=99, idx=0)


def test_top_contexts_is_a_stub_but_validates_pole(tmp_path, vocab):
    _build_fake_mlp_run(tmp_path)
    loader = Loader(tmp_path, _FakeTokenizer(vocab))

    with pytest.raises(ValueError):
        loader.top_contexts(layer=0, idx=0, pole="sideways", n=5)

    with pytest.raises(NotImplementedError):
        loader.top_contexts(layer=0, idx=0, pole="pos", n=5)
