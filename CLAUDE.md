# CLAUDE.md — project notes for coding agents

Read `project_outline.md` (the brief) and `task_list.md` (progress) first. This file records decisions and nuances agreed with the user that are not obvious from those.

## Environment
- Work happens in WSL2 Ubuntu (distro name `Ubuntu`; the default WSL distro on this machine is `docker-desktop`, so always use `wsl -d Ubuntu`). Repo lives at `~/llm_interp` on the WSL filesystem — never `/mnt/c`.
- GPU: RTX 2060, 6144 MiB VRAM, driver 610.88, Turing (cc 7.5). fp32 locally; `sdpa` for capture, `eager` only for attention validation; no flash-attn, no `torch.compile`.
- System python is 3.14; let `uv` manage a 3.11/3.12 interpreter for the project.
- Git user name/email may not be configured in WSL; check before committing.

## Dependency policy
- Pinned: torch, transformers (>=4.51), zarr (v3), pyarrow, duckdb, pydantic, datasets, pytest.
- Ask before adding anything else. Approved by the user: stdlib `argparse` for the CLI. PyYAML/numpy are OK only as transitive/needed-for-YAML dependencies — flag it when adding.
- Pin `Qwen/Qwen3-0.6B` to a commit hash (agent looks it up on the Hub).
- ARC-Easy download (via `datasets`) should be cached under HF_HOME.

## Process rules
- Build order: config -> adapter -> hooks -> checks a/c/e -> generation -> capture/store -> checks b/d/f -> CLI -> notebook. Run tests after each step.
- Ask before changing the storage format.
- No distributed frameworks; no database beyond Parquet + DuckDB.
- Do not implement experiments. Stop at the Definition of Done in the outline.
- Git repo is initialized; the manifest records the commit hash.

## Span / table semantics (user-specified, authoritative)
- Spans are half-open `[start, end)` token positions in the full sequence, keyed by `trace_id` and span name.
- `think`: tokens strictly between `<think>` and `</think>`. Also store `think_open` and `think_close` as one-token spans (the `</think>` position is a likely analysis target).
- `response`: first token after `</think>` up to, not including, `<|im_end|>`/EOS. Keep whitespace tokens; trim at analysis time.
- `answer_value`: sub-span for the token(s) carrying the parsed letter, from the LAST valid JSON `"answer"` in the response. Map character offsets to token positions by incrementally decoding prefixes of the stored IDs; never re-tokenize.
- `hint`: located via the tokenizer offset mapping on the rendered prompt.
- Trace-level `parse_status`: `ok` / `not_found` / `malformed` / `multiple`.
- If `</think>` never appears (truncated generation): no `response` span; flag the trace.
- Table division of labor:
  - `spans`: deterministic facts derived from token IDs (think, think_open, think_close, response, answer_value, hint).
  - `traces`: metadata, including `parsed_answer`, `parse_status`, stop reason, truncation flag, seed.
  - `annotations`: reserved for later judgments (LLM judge, human review); append-only; each row tagged with source and version.
  - `index`: trace_id -> start/end token offsets.

## Technical nuances (from the brief, easy to forget)
- Token IDs are the source of truth. Never re-tokenize decoded text; never re-render via chat template for capture (Qwen3's template strips `<think>` from earlier turns). Render once with `enable_thinking=True`, tokenize once with `add_special_tokens=False`, assert equals `apply_chat_template(tokenize=True)`.
- Qwen3-0.6B: hidden 1024, 16 q heads x head_dim 128 = 2048 (head_dim != hidden/heads), 8 KV heads, tied embeddings, QK-norm before RoPE, theta 1e6, no sliding window. Read all dims from config.
- Don't use `output_hidden_states` (last entry is post-final-norm).
- Sampling is never greedy; cap `max_new_tokens`.
- Position 0 is an attention sink: flag it, exclude from stats by default.
- Captures live on CPU. fp16 is never a substitute for bf16. Never quantize. Never mix fp32 and bf16 captures in one analysis.
- bf16 stored as raw uint16 with a dtype attribute.
- Determinism: fixed seeds, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## VS Code + coding agents in WSL
- Open the folder via `code .` from inside WSL (or "WSL: Connect to WSL" -> Open Folder `/home/luke/llm_interp`). The extension host, terminals, Python, and Claude Code then all run inside WSL.
- Claude Code extension must be installed in the WSL window ("Install in WSL: Ubuntu" on the extension page). The Claude Code CLI should also be installed inside WSL.
- Select the project's `.venv` (created by `uv sync`) as the Python interpreter.
