import json

import pytest

from llm_interp.cli import main

pytestmark = pytest.mark.gpu


def test_generate_capture_inspect_end_to_end(tmp_path, capsys):
    storage_root = str(tmp_path / "run")

    exit_code = main(
        [
            "generate",
            "--config",
            "configs/qwen3_0p6b_local.yaml",
            "--num-pairs",
            "1",
            "--storage-root",
            storage_root,
        ]
    )
    assert exit_code == 0
    assert "Generated 2 traces (1 pairs)" in capsys.readouterr().out
    assert (tmp_path / "run" / "generated_traces.json").exists()

    exit_code = main(["capture", "--config", "configs/qwen3_0p6b_local.yaml", "--storage-root", storage_root])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Captured 2 traces" in captured
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "index.parquet").exists()
    assert (tmp_path / "run" / "activations.zarr").exists()

    exit_code = main(["inspect", "--config", "configs/qwen3_0p6b_local.yaml", "--storage-root", storage_root])
    assert exit_code == 0
    listing = capsys.readouterr().out
    lines = [line for line in listing.strip().splitlines() if line]
    assert len(lines) == 2  # no_hint + hint traces for the one pair
    trace_id = lines[0].split("\t")[0]

    exit_code = main(
        ["inspect", "--config", "configs/qwen3_0p6b_local.yaml", "--storage-root", storage_root, "--trace-id", trace_id]
    )
    assert exit_code == 0
    detail = capsys.readouterr().out
    assert "spans:" in detail
    assert "tokens:" in detail
    meta_json = detail.split("\n\nspans:")[0]
    meta = json.loads(meta_json)
    assert meta["trace_id"] == trace_id


def test_capture_without_generate_fails_cleanly(tmp_path, capsys):
    storage_root = str(tmp_path / "empty_run")
    exit_code = main(["capture", "--config", "configs/qwen3_0p6b_local.yaml", "--storage-root", storage_root])
    assert exit_code == 1
    assert "Run `generate` first" in capsys.readouterr().err


def test_doctor_via_cli_main():
    exit_code = main(["doctor", "--config", "configs/qwen3_0p6b_local.yaml"])
    assert exit_code == 0


def test_estimate_via_cli_main(capsys):
    exit_code = main(["estimate", "--config", "configs/qwen3_0p6b_local.yaml", "--num-traces", "10"])
    assert exit_code == 0
    assert "projected total" in capsys.readouterr().out
