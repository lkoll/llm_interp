from pathlib import Path

import torch

from .adapters.base import ArchAdapter
from .config import RunConfig
from .hooks import HookManager
from .prompts.pairs import PairTrace
from .recompute import recompute_qk_pre_rope
from .spans import find_answer_value_span, find_hint_span, find_response_span, find_think_spans
from .store import TraceTables, ZarrStore

TRACE_ID_SEP = "__"


def trace_id_for(pair_id: str, condition: str) -> str:
    return f"{pair_id}{TRACE_ID_SEP}{condition}"


def _eos_token_ids(model: torch.nn.Module) -> set[int]:
    eos = model.config.eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


class CaptureRunner:
    """Teacher-forced capture over already-generated traces (`PairTrace`, from
    `prompts.generate_pair`). Generation and capture are separate stages that
    talk only through the trace store — this takes token IDs in, replays them
    teacher-forced in one forward pass (`use_cache=False`, batch size 1), and
    writes resid_pre/final_residual/top-k logprobs plus the derived spans.

    Resumable: on `run()`, the Zarr arrays are first truncated back to the last
    indexed offset (discarding any data from a trace that crashed mid-write),
    then any trace already present in `index` is skipped.
    """

    def __init__(self, config: RunConfig, model: torch.nn.Module, tokenizer, adapter: ArchAdapter, store_root: str | Path):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.store_root = Path(store_root)
        self.zarr_store = ZarrStore(self.store_root / "activations.zarr")
        self.tables = TraceTables(self.store_root)
        self.think_open_id = tokenizer.convert_tokens_to_ids("<think>")
        self.think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        self.eos_ids = _eos_token_ids(model)

    def run(self, pair_traces: list[PairTrace]) -> None:
        self.zarr_store.truncate_all(self.tables.last_index_end())
        done = self.tables.done_trace_ids()
        for trace in pair_traces:
            trace_id = trace_id_for(trace.record.pair_id, trace.record.condition)
            if trace_id in done:
                continue
            self._capture_one(trace_id, trace)

    def _capture_one(self, trace_id: str, trace: PairTrace) -> None:
        full_ids_cpu = trace.generation_result.full_ids  # 1D LongTensor, CPU
        full_ids_list = full_ids_cpu.tolist()
        n_tokens = len(full_ids_list)
        device = next(self.model.parameters()).device
        device_ids = full_ids_cpu.unsqueeze(0).to(device)

        mgr = HookManager(self.adapter).capture_resid_pre(self.config.capture.layers)
        if self.config.capture.capture_final_residual:
            mgr.capture_final_residual()
        with mgr:
            with torch.no_grad():
                out = self.model(input_ids=device_ids, use_cache=False)

        logits = out.logits[0].float()  # [N, vocab] — full-vocab logits, never stored
        log_probs = torch.log_softmax(logits, dim=-1)
        top_k = self.config.capture.top_k_logprobs
        top_values, top_indices = log_probs.topk(top_k, dim=-1)

        realized_logprob = torch.full((n_tokens,), float("nan"), device=log_probs.device)
        if n_tokens > 1:
            realized_logprob[:-1] = (
                log_probs[:-1].gather(1, device_ids[0, 1:].unsqueeze(1)).squeeze(1)
            )

        self._write_sites(mgr, n_tokens, top_values, top_indices, realized_logprob)
        start, end = self.zarr_store.append("positions", torch.arange(n_tokens))

        span_rows = self._compute_span_rows(trace_id, trace, full_ids_list)

        self.tables.add_spans_rows(span_rows)
        self.tables.add_traces_row(
            {
                "trace_id": trace_id,
                "pair_id": trace.record.pair_id,
                "condition": trace.record.condition,
                "hinted_option": trace.record.hinted_option,
                "gold": trace.record.gold,
                "parsed_answer": trace.record.parsed_answer,
                "parse_status": trace.record.parse_status,
                "stop_reason": trace.generation_result.stop_reason,
                "truncated": trace.generation_result.truncated,
                "seed": trace.generation_result.seed,
                "prompt_len": trace.generation_result.prompt_len,
                "total_len": n_tokens,
                "has_think_close": any(r["span_name"] == "think_close" for r in span_rows),
            }
        )
        self.tables.add_index_row({"trace_id": trace_id, "start": start, "end": end})
        self.tables.flush()

    def _write_sites(
        self,
        mgr: HookManager,
        n_tokens: int,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
        realized_logprob: torch.Tensor,
    ) -> None:
        # "positions" (appended by the caller) is the canonical (start, end) for
        # this trace; every other required site must agree with it exactly.
        canonical: tuple[int, int] | None = None
        for layer in self.config.capture.layers:
            offset = self.zarr_store.append(f"resid_pre/{layer}", mgr.captures[layer][0])
            canonical = canonical or offset
            assert offset == canonical, f"resid_pre/{layer} offset {offset} != {canonical}"
        if self.config.capture.capture_final_residual:
            offset = self.zarr_store.append("final_residual", mgr.captures["final_residual"][0])
            canonical = canonical or offset
            assert offset == canonical, f"final_residual offset {offset} != {canonical}"
        for site, data in (
            ("top_logprob_values", top_values.to("cpu")),
            ("top_logprob_indices", top_indices.to("cpu").to(torch.long)),
            ("realized_logprob", realized_logprob.to("cpu")),
        ):
            offset = self.zarr_store.append(site, data)
            canonical = canonical or offset
            assert offset == canonical, f"{site} offset {offset} != {canonical}"

        if self.config.capture.capture_qk_pre_rope:
            for layer in self.config.capture.layers:
                q, k = recompute_qk_pre_rope(self.adapter, mgr.captures[layer], layer)
                self.zarr_store.append(f"qk_pre_rope/{layer}_q", q[0].transpose(0, 1).contiguous().to("cpu"))
                self.zarr_store.append(f"qk_pre_rope/{layer}_k", k[0].transpose(0, 1).contiguous().to("cpu"))

    def _compute_span_rows(self, trace_id: str, trace: PairTrace, full_ids_list: list[int]) -> list[dict]:
        span_rows: list[dict] = []

        think_spans = find_think_spans(full_ids_list, self.think_open_id, self.think_close_id)
        if think_spans is not None:
            for name in ("think_open", "think", "think_close"):
                span = think_spans[name]
                span_rows.append({"trace_id": trace_id, "span_name": name, "start": span.start, "end": span.end})

            response_span = find_response_span(full_ids_list, think_spans["think_close"].end, self.eos_ids)
            span_rows.append(
                {"trace_id": trace_id, "span_name": "response", "start": response_span.start, "end": response_span.end}
            )

            answer_value_span = find_answer_value_span(self.tokenizer, full_ids_list, response_span)
            if answer_value_span is not None:
                span_rows.append(
                    {
                        "trace_id": trace_id,
                        "span_name": "answer_value",
                        "start": answer_value_span.start,
                        "end": answer_value_span.end,
                    }
                )

        hint_span = find_hint_span(trace.rendered_prompt, trace.record.hinted_option)
        if hint_span is not None:
            span_rows.append(
                {"trace_id": trace_id, "span_name": "hint", "start": hint_span.start, "end": hint_span.end}
            )

        return span_rows
