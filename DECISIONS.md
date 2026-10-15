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

## 2026-10-15 — RunConfig shape

`RunConfig` (`src/llm_interp/config/schema.py`) groups fields into `model` / `capture` / `generation` plus top-level `run_name` and `storage_root`, matching the brief's listed fields (model id+revision, dtype, device, attention impl, layers/sites, storage root, generation params) one-for-one. All four pydantic models are frozen.

- `capture` doesn't carry a generic `sites: list[str]`. The outline's actual capture semantics (step 5) are: resid_pre for the configured `layers`, the final residual, and top-k logprobs, always; post-QK-norm pre-RoPE q/k only for validation runs. Modeled that directly as `layers: list[int]`, `capture_final_residual: bool`, `top_k_logprobs: int`, `capture_qk_pre_rope: bool` rather than inventing a generic site-name list — avoids a layer of indirection nothing downstream needs yet.
- `layers` validation (non-negative, no duplicates, sorted) happens at the schema level; validation that the layer indices are in-range for the actual model (0-27 for the 0.6B, 0-35 for the 8B) is deferred to the adapter once the model config is loaded, per the outline's "read all dims from config" principle — `RunConfig` itself has no model-specific knowledge.
- `generation.max_new_tokens` set to 768 in both configs — not specified in the brief beyond "cap it, 0.6B rambles." A judgment-call default, not validated against actual generation behavior yet; revisit once `estimate`/`generate` exist and real trace lengths are visible.
- Pinned `Qwen/Qwen3-8B` to commit `b968826d9c46dd6066d109eabc6255188de91218` for the (unrun) cloud config, with the same due diligence as the 0.6B pin — fetched `config.json` at that commit and confirmed 36 layers, 32 attention heads, 8 KV heads, head_dim 128, matching the outline.

## 2026-10-15 — ArchAdapter interface and Qwen3Adapter

Read the installed `transformers==5.19.0` source (`transformers/models/qwen3/modeling_qwen3.py` and `configuration_qwen3.py`) directly rather than assuming module names. Confirmed: `model.model.embed_tokens`, `model.model.layers[l]` (resid_pre[l] hook target), `.input_layernorm`, `.self_attn.{q,k,v}_proj`, `.self_attn.{q,k}_norm` (per-head RMSNorm, applied post-reshape pre-RoPE), one shared `model.model.rotary_emb` across all layers, `model.model.norm` (final norm), `model.lm_head`. Per-layer attention type comes from `config.layer_types[layer_idx]` (all `"full_attention"` for Qwen3 — no sliding window).

- `ArchAdapter` (`src/llm_interp/adapters/base.py`) exposes dims as properties (all read from `model.config`, nothing hardcoded) plus one method per logical site, typed to return `nn.Module | None` so a future adapter (Gemma 3, Qwen3.6) can report an absent site. Three recompute primitives live on the adapter rather than in a generic `recompute.py` function table: `reshape_to_heads` (generic default, overridable), `apply_qk_norm` and `apply_rotary` (both abstract — architecture-specific). `recompute.py` (0.4) will orchestrate these plus the GQA repeat and causal-mask attention-score math, which is *not* architecture-specific and doesn't belong on the adapter.
- `apply_rotary` calls the real `apply_rotary_pos_emb` function imported from `modeling_qwen3`, not a reimplementation — guarantees bit-exact RoPE application without duplicating the rotate-half math.
- Verified adapter correctness against the real model, not just shape: `tests/unit/test_adapters.py::test_apply_qk_norm_and_rotary_match_real_forward` monkeypatches the module-level `apply_rotary_pos_emb` during a real forward pass to capture its pre/post-rotary tensors as ground truth (there's no other hook point for that intermediate value), and asserts the adapter's `reshape_to_heads -> apply_qk_norm -> apply_rotary` chain reproduces it bit-for-bit. This is effectively check (c) run early, at the adapter level.
- Tiny random-init Qwen3 fixture (`tests/unit/conftest.py`) deliberately sets `head_dim=16` while `hidden_size=32, num_attention_heads=4` (so `hidden_size // num_attention_heads == 8 != 16`), to exercise the head_dim-!=-hidden/heads gotcha on every adapter test, not just the real 0.6B model.
