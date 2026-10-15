import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..generate import GenerationResult, RenderedPrompt
from .pairs import PairRecord, PairTrace


def save_pair_traces(path: str | Path, traces: list[PairTrace]) -> None:
    """Generation and capture are separate stages that talk only through the
    trace store (design principle 3) — this is that handoff: generation writes
    the full, already-generated traces (token IDs included, the source of
    truth) to a plain JSON file; a later, separate `capture` invocation reads
    it back and replays the token IDs teacher-forced. Not one of the four
    official Parquet tables — just a simple intermediate artifact between two
    CLI steps, not a new database.
    """
    data = []
    for t in traces:
        data.append(
            {
                "record": asdict(t.record),
                "rendered_prompt": {
                    "text": t.rendered_prompt.text,
                    "input_ids": t.rendered_prompt.input_ids[0].tolist(),
                    "offset_mapping": [list(pair) for pair in t.rendered_prompt.offset_mapping],
                },
                "generation_result": {
                    "full_ids": t.generation_result.full_ids.tolist(),
                    "prompt_len": t.generation_result.prompt_len,
                    "stop_reason": t.generation_result.stop_reason,
                    "truncated": t.generation_result.truncated,
                    "seed": t.generation_result.seed,
                },
            }
        )
    Path(path).write_text(json.dumps(data, indent=2))


def load_pair_traces(path: str | Path) -> list[PairTrace]:
    data = json.loads(Path(path).read_text())
    traces = []
    for item in data:
        record = PairRecord(**item["record"])
        rp = item["rendered_prompt"]
        rendered_prompt = RenderedPrompt(
            text=rp["text"],
            input_ids=torch.tensor([rp["input_ids"]], dtype=torch.long),
            offset_mapping=[tuple(pair) for pair in rp["offset_mapping"]],
        )
        gr = item["generation_result"]
        generation_result = GenerationResult(
            full_ids=torch.tensor(gr["full_ids"], dtype=torch.long),
            prompt_len=gr["prompt_len"],
            stop_reason=gr["stop_reason"],
            truncated=gr["truncated"],
            seed=gr["seed"],
        )
        traces.append(PairTrace(record=record, rendered_prompt=rendered_prompt, generation_result=generation_result))
    return traces
