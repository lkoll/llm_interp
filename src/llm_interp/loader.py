from dataclasses import dataclass
from pathlib import Path

import torch

from .mlp import regime_from_bits, unpack_gate_sign_bits
from .store import TraceTables, ZarrStore


@dataclass(frozen=True)
class LoadedTensor:
    tensor: torch.Tensor
    token_ids: list[int]
    decoded_tokens: list[str]


@dataclass(frozen=True)
class NeuronValues:
    """One MLP neuron's values/regimes across a token span (Addendum A).
    `available` is False wherever the value wasn't kept by topk storage for
    that layer — `values`/`regime` carry sentinels (NaN / 255) there, not real
    data."""

    values: torch.Tensor  # [span_len], float
    regime: torch.Tensor  # [span_len], uint8 (255 = unavailable)
    available: torch.Tensor  # [span_len], bool


class Loader:
    """(trace_id, site, layer, token span) -> tensor, returned with decoded
    tokens. Reads a completed (or in-progress but already-flushed) run; opens the
    store and tables once at construction, not refreshed afterward."""

    def __init__(self, store_root: str | Path, tokenizer):
        self.store_root = Path(store_root)
        self.zarr_store = ZarrStore(self.store_root / "activations.zarr")
        self.tables = TraceTables(self.store_root)
        self.tokenizer = tokenizer

        self._index: dict[str, tuple[int, int]] = {
            row["trace_id"]: (row["start"], row["end"]) for row in self.tables._rows["index"]
        }
        self._traces_by_id: dict[str, dict] = {row["trace_id"]: row for row in self.tables._rows["traces"]}
        self._spans_by_id: dict[str, list[dict]] = {}
        for row in self.tables._rows["spans"]:
            self._spans_by_id.setdefault(row["trace_id"], []).append(row)

    def trace_ids(self) -> list[str]:
        return list(self._index.keys())

    def trace_metadata(self, trace_id: str) -> dict:
        return self._traces_by_id[trace_id]

    def spans_for(self, trace_id: str) -> list[dict]:
        return self._spans_by_id.get(trace_id, [])

    def _resolve_span(self, trace_id: str, span: tuple[int, int] | None) -> tuple[int, int]:
        if trace_id not in self._index:
            raise KeyError(f"unknown trace_id: {trace_id!r}")
        trace_start, trace_end = self._index[trace_id]
        trace_len = trace_end - trace_start

        if span is None:
            span = (0, trace_len)
        span_start, span_end = span
        if not (0 <= span_start <= span_end <= trace_len):
            raise ValueError(f"span {span} out of bounds for trace {trace_id!r} of length {trace_len}")

        return trace_start + span_start, trace_start + span_end

    def load(
        self,
        trace_id: str,
        site: str,
        layer: int | None = None,
        span: tuple[int, int] | None = None,
    ) -> LoadedTensor:
        abs_start, abs_end = self._resolve_span(trace_id, span)

        site_path = f"{site}/{layer}" if layer is not None else site
        tensor = self.zarr_store.read(site_path, abs_start, abs_end)
        token_ids = self.zarr_store.read("token_ids", abs_start, abs_end).tolist()
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        return LoadedTensor(tensor=tensor, token_ids=token_ids, decoded_tokens=decoded_tokens)

    def get_neuron(
        self, trace_id: str, layer: int, idx: int, span: tuple[int, int] | None = None
    ) -> NeuronValues:
        """One MLP neuron's values + regime codes across `span` (Addendum A).
        Works against either storage mode, transparently: dense (`mlp_act` +
        `mlp_gate_sign`, exact for every position) or topk (`mlp_topk_*`, exact
        only where `idx` was among the kept entries for that token — elsewhere
        `available` is False)."""
        abs_start, abs_end = self._resolve_span(trace_id, span)
        n = abs_end - abs_start

        if f"mlp_act/{layer}" in self.zarr_store.group:
            act = self.zarr_store.read(f"mlp_act/{layer}", abs_start, abs_end)  # [n, d_mlp]
            gate_sign_packed = self.zarr_store.read(f"mlp_gate_sign/{layer}", abs_start, abs_end).numpy()
            gate_sign = torch.from_numpy(unpack_gate_sign_bits(gate_sign_packed, d_mlp=act.shape[-1])).bool()
            values = act[:, idx]
            regime = regime_from_bits(gate_sign[:, idx], values)
            return NeuronValues(values=values, regime=regime, available=torch.ones(n, dtype=torch.bool))

        if f"mlp_topk_index/{layer}" in self.zarr_store.group:
            indices = self.zarr_store.read(f"mlp_topk_index/{layer}", abs_start, abs_end)  # [n, k]
            topk_values = self.zarr_store.read(f"mlp_topk_value/{layer}", abs_start, abs_end)
            topk_regime = self.zarr_store.read(f"mlp_topk_regime/{layer}", abs_start, abs_end)

            values = torch.full((n,), float("nan"))
            regime = torch.full((n,), 255, dtype=torch.uint8)
            available = torch.zeros(n, dtype=torch.bool)

            matches = indices == idx  # [n, k]
            rows_with_match = matches.any(dim=-1)
            match_col = matches.float().argmax(dim=-1)
            values[rows_with_match] = topk_values[rows_with_match, match_col[rows_with_match]]
            regime[rows_with_match] = topk_regime[rows_with_match, match_col[rows_with_match]]
            available[rows_with_match] = True
            return NeuronValues(values=values, regime=regime, available=available)

        raise KeyError(f"no MLP capture (dense or topk) found for layer {layer}")

    def top_contexts(self, layer: int, idx: int, pole: str, n: int, context_tokens: int = 1024) -> list:
        """Interface stub only (Addendum A) — Phase 1 builds the actual index of
        each neuron's strongest activations across a run. `pole` in
        {"pos", "neg"}: both carry meaning (Addendum A, from [1]), built
        separately. Must return generous preceding context (>= 1,024 tokens by
        default), not short snippets — short context makes long-context
        neurons look polysemantic."""
        if pole not in ("pos", "neg"):
            raise ValueError(f"pole must be 'pos' or 'neg', got {pole!r}")
        raise NotImplementedError(
            "top_contexts is a Phase 1 stub: building the per-neuron strongest-activation "
            "index across a full run is out of scope for Phase 0 (Addendum A)."
        )
