# Decisions

Append-only log of design choices not fully specified in `project_outline.md`, in the order they were made. Each entry: date, decision, why.

## 2026-10-15 — Dependency versions

Pinned exact versions (not ranges) for all approved dependencies, resolved against PyPI latest at the time:
torch 2.14.1, transformers 5.19.0, zarr 3.4.1, pyarrow 26.0.0, duckdb 1.5.6, pydantic 2.14.1, datasets 5.1.0, pytest 9.1.1 (dev group).

- `transformers`: the brief said `>=4.51`. By now PyPI has moved past a major version bump (last 4.x was 4.57.6, latest is 5.19.0, 39 releases into 5.x). Asked the user; chose latest 5.19.0 over the last 4.x release.
- `torch`: plain `torch==2.14.1` from PyPI, no custom CUDA index URL. Checked its `requires_dist`: it bundles CUDA 13.0.3 via `nvidia-*`/`cuda-toolkit` wheels. The WSL GPU driver reports `CUDA UMD Version: 13.3`, which is forward-compatible with 13.0.3. Turing (sm_75, RTX 2060) is still a supported architecture at CUDA 13. To be confirmed empirically by `doctor` once built.
- `pyyaml==6.0.3` added as a direct dependency (flagged per CLAUDE.md's dependency policy) — needed to parse the YAML run configs. `numpy` was deliberately *not* added directly; it will resolve transitively via torch/pyarrow/zarr/datasets.

## 2026-10-15 — Qwen3-0.6B pin

Pinned `Qwen/Qwen3-0.6B` to commit `c1899de289a04d12100db370d81485cdf75e47ca` (last modified 2025-07-26, via the HF Hub API). Fetched `config.json` at that commit and confirmed it matches the dims recorded in CLAUDE.md: hidden_size 1024, num_attention_heads 16, head_dim 128 (16*128=2048 != hidden_size, as flagged), num_key_value_heads 8, num_hidden_layers 28, tied embeddings, rope_theta 1e6, no sliding window.

## 2026-10-15 — Package layout

Src-layout: `src/llm_interp/` as the installable package (hatchling build backend), with subpackages `config/`, `adapters/`, `generate/`, `prompts/`, `store/`, `validate/` matching the outline's module list; `hooks.py`, `capture.py`, `recompute.py`, `loader.py`, `cli.py` live as flat modules directly under `src/llm_interp/` (added as each build step is implemented, not pre-created as empty stubs). `configs/` (YAML run configs) and `tests/{unit,integration}/` sit at the repo root, not inside the package. Chosen for standard, uv-friendly packaging and clean `pytest`/CLI imports; not specified explicitly in the brief.

## 2026-10-15 — Python interpreter

`requires-python = ">=3.12,<3.13"`, pinned via `.python-version` to a uv-managed CPython 3.12.15 (not the system Python, which is 3.14). Narrowed from the initially-planned `>=3.11` after `zarr==3.4.1` turned out to require Python >=3.12 — still within CLAUDE.md's "let uv manage a 3.11/3.12 interpreter."
