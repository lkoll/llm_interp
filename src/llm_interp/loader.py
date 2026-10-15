from dataclasses import dataclass
from pathlib import Path

import torch

from .store import TraceTables, ZarrStore


@dataclass(frozen=True)
class LoadedTensor:
    tensor: torch.Tensor
    token_ids: list[int]
    decoded_tokens: list[str]


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

    def load(
        self,
        trace_id: str,
        site: str,
        layer: int | None = None,
        span: tuple[int, int] | None = None,
    ) -> LoadedTensor:
        if trace_id not in self._index:
            raise KeyError(f"unknown trace_id: {trace_id!r}")
        trace_start, trace_end = self._index[trace_id]
        trace_len = trace_end - trace_start

        if span is None:
            span = (0, trace_len)
        span_start, span_end = span
        if not (0 <= span_start <= span_end <= trace_len):
            raise ValueError(f"span {span} out of bounds for trace {trace_id!r} of length {trace_len}")

        abs_start = trace_start + span_start
        abs_end = trace_start + span_end

        site_path = f"{site}/{layer}" if layer is not None else site
        tensor = self.zarr_store.read(site_path, abs_start, abs_end)
        token_ids = self.zarr_store.read("token_ids", abs_start, abs_end).tolist()
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        return LoadedTensor(tensor=tensor, token_ids=token_ids, decoded_tokens=decoded_tokens)
