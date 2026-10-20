import os

# Must be set before the first CUDA context init for deterministic cuBLAS algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from pathlib import Path

import torch

from ..config import load_config
from ..estimate import default_tokens_per_trace
from ..model_loading import build_adapter, load_model_and_tokenizer
from .base import CheckResult, ValidationContext
from .checks import (
    check_a_logits_from_final_residual,
    check_b_kv_cache_vs_teacher_forced,
    check_c_recomputed_qk,
    check_d_attention_probs_vs_eager,
    check_e_storage_round_trip,
    check_f_add_vector,
    check_g_mlp_consistency,
)
from .cloud_checks import (
    check_cloud_estimate_vs_free_space,
    check_cloud_gpu_report,
    check_cloud_network_volume,
    check_cloud_storage_throughput,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "qwen3_0p6b_local.yaml"
VALIDATION_TEXT = "The quick brown fox jumps over the lazy dog, and the five boxing wizards jump quickly."


def print_report(results: list[CheckResult]) -> None:
    name_width = max(len(r.name) for r in results) + 2
    print(f"{'CHECK'.ljust(name_width)}STATUS")
    print("-" * (name_width + 6))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name.ljust(name_width)}{status}")
        if r.error:
            print(f"    error: {r.error}")
        else:
            detail_str = ", ".join(f"{k}={v}" for k, v in r.details.items() if not isinstance(v, dict))
            if detail_str:
                print(f"    {detail_str}")
    n_passed = sum(r.passed for r in results)
    print(f"\n{n_passed}/{len(results)} checks passed.")


def run_doctor(
    config_path: str | Path,
    num_traces: int = 60,
    tokens_per_trace: int | None = None,
    storage_root: str | Path | None = None,
) -> list[CheckResult]:
    config = load_config(config_path)
    if storage_root is not None:
        config = config.model_copy(update={"storage_root": Path(storage_root)})

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model, tokenizer = load_model_and_tokenizer(config)
    adapter = build_adapter(model)
    ctx = ValidationContext(config=config, model=model, tokenizer=tokenizer, adapter=adapter)
    input_ids = tokenizer(VALIDATION_TEXT, return_tensors="pt").input_ids

    if tokens_per_trace is None:
        tokens_per_trace = default_tokens_per_trace(config)
    num_tokens = num_traces * tokens_per_trace

    return [
        check_a_logits_from_final_residual(ctx, input_ids),
        check_b_kv_cache_vs_teacher_forced(ctx, input_ids[0]),
        check_c_recomputed_qk(ctx, input_ids),
        check_d_attention_probs_vs_eager(ctx, input_ids),
        check_e_storage_round_trip(),
        check_f_add_vector(ctx, input_ids),
        check_g_mlp_consistency(ctx, input_ids),
        # Addendum B: skipped (reported as passed) unless RUNPOD_POD_ID is set --
        # see cloud_checks.py's module docstring.
        check_cloud_gpu_report(config),
        check_cloud_network_volume(config),
        check_cloud_storage_throughput(config),
        check_cloud_estimate_vs_free_space(config, num_tokens),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all validation checks (a-g, plus cloud checks on a RunPod pod) and print a pass/fail table.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a RunConfig YAML file.")
    parser.add_argument("--num-traces", type=int, default=60, help="For the cloud estimate-vs-free-space check.")
    parser.add_argument("--tokens-per-trace", type=int, default=None)
    parser.add_argument("--storage-root", default=None, help="Overrides the config's storage_root.")
    args = parser.parse_args(argv)

    results = run_doctor(
        args.config,
        num_traces=args.num_traces,
        tokens_per_trace=args.tokens_per_trace,
        storage_root=args.storage_root,
    )
    print_report(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
