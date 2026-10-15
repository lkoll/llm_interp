from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INDEX_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
    ]
)

TRACES_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("pair_id", pa.string()),
        ("condition", pa.string()),
        ("hinted_option", pa.string()),
        ("gold", pa.string()),
        ("parsed_answer", pa.string()),
        ("parse_status", pa.string()),
        ("stop_reason", pa.string()),
        ("truncated", pa.bool_()),
        ("seed", pa.int64()),
        ("prompt_len", pa.int64()),
        ("total_len", pa.int64()),
        ("has_think_close", pa.bool_()),
    ]
)

SPANS_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("span_name", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
    ]
)

# Reserved for later judgments (LLM judge, human review); append-only, unused in Phase 0.
ANNOTATIONS_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("source", pa.string()),
        ("version", pa.string()),
        ("key", pa.string()),
        ("value", pa.string()),
    ]
)

_FILES = {
    "index": "index.parquet",
    "traces": "traces.parquet",
    "spans": "spans.parquet",
    "annotations": "annotations.parquet",
}
_SCHEMAS = {
    "index": INDEX_SCHEMA,
    "traces": TRACES_SCHEMA,
    "spans": SPANS_SCHEMA,
    "annotations": ANNOTATIONS_SCHEMA,
}


class TraceTables:
    """The four Parquet tables (index, traces, spans, annotations), held in
    memory and rewritten whole on each `flush()` — simple and safe at this
    project's toy scale (tens of traces), where true incremental Parquet
    appending would add complexity for no real benefit.

    Resumable: on load, any `traces`/`spans` rows whose `trace_id` isn't in
    `index` are dropped — orphaned writes from a run that crashed between
    flushing those and flushing `index` (the recovery signal: a trace only
    counts as done once it's in `index`).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, list[dict]] = {name: self._load(name) for name in _FILES}
        done_ids = {row["trace_id"] for row in self._rows["index"]}
        self._rows["traces"] = [r for r in self._rows["traces"] if r["trace_id"] in done_ids]
        self._rows["spans"] = [r for r in self._rows["spans"] if r["trace_id"] in done_ids]

    def _path(self, name: str) -> Path:
        return self.root / _FILES[name]

    def _load(self, name: str) -> list[dict]:
        path = self._path(name)
        if path.exists():
            return pq.read_table(path).to_pylist()
        return []

    def done_trace_ids(self) -> set[str]:
        return {row["trace_id"] for row in self._rows["index"]}

    def last_index_end(self) -> int:
        if not self._rows["index"]:
            return 0
        return max(row["end"] for row in self._rows["index"])

    def add_traces_row(self, row: dict) -> None:
        self._rows["traces"].append(row)

    def add_spans_rows(self, rows: list[dict]) -> None:
        self._rows["spans"].extend(rows)

    def add_index_row(self, row: dict) -> None:
        self._rows["index"].append(row)

    def flush(self) -> None:
        for name in _FILES:
            table = pa.Table.from_pylist(self._rows[name], schema=_SCHEMAS[name])
            pq.write_table(table, self._path(name))
