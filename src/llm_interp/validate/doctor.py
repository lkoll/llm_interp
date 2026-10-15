import os

# Must be set before the first CUDA context init for deterministic cuBLAS algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..adapters import Qwen3Adapter
from ..config import load_config
from .base import CheckResult, ValidationContext
from .checks import (
    check_a_logits_from_final_residual,
    check_b_kv_cache_vs_teacher_forced,
    check_c_recomputed_qk,
    check_d_attention_probs_vs_eager,
    check_e_storage_round_trip,
    check_f_add_vector,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "qwen3_0p6b_local.yaml"
VALIDATION_TEXT = "The quick brown fox jumps over the lazy dog, and the five boxing wizards jump quickly."

_TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


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


def run_doctor(config_path: str | Path) -> list[CheckResult]:
    config = load_config(config_path)

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=_TORCH_DTYPES[config.model.dtype],
        attn_implementation=config.model.attn_implementation,
    )
    model = model.to(config.model.device)
    model.eval()

    adapter = Qwen3Adapter(model)
    ctx = ValidationContext(config=config, model=model, tokenizer=tokenizer, adapter=adapter)
    input_ids = tokenizer(VALIDATION_TEXT, return_tensors="pt").input_ids

    return [
        check_a_logits_from_final_residual(ctx, input_ids),
        check_b_kv_cache_vs_teacher_forced(ctx, input_ids[0]),
        check_c_recomputed_qk(ctx, input_ids),
        check_d_attention_probs_vs_eager(ctx, input_ids),
        check_e_storage_round_trip(),
        check_f_add_vector(ctx, input_ids),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all validation checks (a-f) and print a pass/fail table.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a RunConfig YAML file.")
    args = parser.parse_args(argv)

    results = run_doctor(args.config)
    print_report(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
