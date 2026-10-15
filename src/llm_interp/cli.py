import argparse
import json
import sys
from pathlib import Path

from .capture import CaptureRunner
from .config import load_config
from .estimate import estimate_bytes, print_estimate
from .generate import HFGenerator
from .loader import Loader
from .model_loading import build_adapter, load_model_and_tokenizer
from .prompts import ArcEasyPromptSource, generate_pair, load_pair_traces, save_pair_traces
from .validate.doctor import DEFAULT_CONFIG_PATH, print_report, run_doctor

GENERATED_TRACES_FILENAME = "generated_traces.json"


def cmd_doctor(args: argparse.Namespace) -> int:
    results = run_doctor(args.config)
    print_report(results)
    return 0 if all(r.passed for r in results) else 1


def cmd_estimate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tokens_per_trace = args.tokens_per_trace
    if tokens_per_trace is None:
        tokens_per_trace = config.generation.max_new_tokens + 100  # nominal prompt length
    num_tokens = args.num_traces * tokens_per_trace
    result = estimate_bytes(config, num_tokens)
    print_estimate(result)
    return 0


def _storage_root(config, args: argparse.Namespace) -> Path:
    return Path(args.storage_root) if args.storage_root else Path(config.storage_root)


def cmd_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(config)
    generator = HFGenerator(model)

    items = ArcEasyPromptSource(n=args.num_pairs).load()
    all_traces = []
    for item in items:
        all_traces.extend(generate_pair(item, generator, tokenizer, config.generation))

    out_path = _storage_root(config, args) / GENERATED_TRACES_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pair_traces(out_path, all_traces)
    print(f"Generated {len(all_traces)} traces ({len(items)} pairs) -> {out_path}")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(config)
    adapter = build_adapter(model)
    storage_root = _storage_root(config, args)

    traces_path = storage_root / GENERATED_TRACES_FILENAME
    if not traces_path.exists():
        print(f"No generated traces found at {traces_path}. Run `generate` first.", file=sys.stderr)
        return 1
    traces = load_pair_traces(traces_path)

    runner = CaptureRunner(config, model, tokenizer, adapter, storage_root)
    runner.run(traces)
    print(f"Captured {len(traces)} traces -> {storage_root}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)
    loader = Loader(_storage_root(config, args), tokenizer)

    if args.trace_id is None:
        for trace_id in sorted(loader.trace_ids()):
            meta = loader.trace_metadata(trace_id)
            print(f"{trace_id}\tcondition={meta['condition']}\tparsed={meta['parsed_answer']}\tstatus={meta['parse_status']}")
        return 0

    meta = loader.trace_metadata(args.trace_id)
    print(json.dumps(meta, indent=2))
    print("\nspans:")
    for span in loader.spans_for(args.trace_id):
        print(f"  {span['span_name']}: [{span['start']}, {span['end']})")
    loaded = loader.load(args.trace_id, "token_ids")
    print("\ntokens:")
    print(" ".join(loaded.decoded_tokens))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-interp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_doctor = subparsers.add_parser("doctor", help="Run validation checks a-f and print a pass/fail table.")
    p_doctor.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_doctor.set_defaults(func=cmd_doctor)

    p_estimate = subparsers.add_parser("estimate", help="Estimate storage bytes for a capture run before running it.")
    p_estimate.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_estimate.add_argument("--num-traces", type=int, default=60, help="e.g. 30 pairs x 2 conditions.")
    p_estimate.add_argument("--tokens-per-trace", type=int, default=None)
    p_estimate.set_defaults(func=cmd_estimate)

    p_generate = subparsers.add_parser("generate", help="Generate paired-hint traces and save them for `capture`.")
    p_generate.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_generate.add_argument("--num-pairs", type=int, default=30)
    p_generate.add_argument("--storage-root", default=None, help="Overrides the config's storage_root.")
    p_generate.set_defaults(func=cmd_generate)

    p_capture = subparsers.add_parser("capture", help="Teacher-forced capture over traces from `generate`.")
    p_capture.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_capture.add_argument("--storage-root", default=None, help="Overrides the config's storage_root.")
    p_capture.set_defaults(func=cmd_capture)

    p_inspect = subparsers.add_parser("inspect", help="List traces, or show one trace's metadata/spans/tokens.")
    p_inspect.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_inspect.add_argument("--storage-root", default=None, help="Overrides the config's storage_root.")
    p_inspect.add_argument("--trace-id", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
