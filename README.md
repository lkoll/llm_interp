# llm_interp

Activation capture pipeline for interpretability research on LLM reasoning (Phase 0: Qwen3-0.6B, local GPU). See `project_outline.md` for the full brief, `task_list.md` for progress, and `CLAUDE.md` / `DECISIONS.md` for agreed conventions and design decisions.

## Setup (WSL2 Ubuntu)

Prerequisites: WSL2 Ubuntu with GPU passthrough (`nvidia-smi` should show your GPU), repo checked out on the WSL filesystem (not `/mnt/c`), [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv sync
```

This creates `.venv` using a uv-managed Python 3.12 (pinned via `.python-version`), independent of the system Python.

For the notebook (plotting, Jupyter kernel), also sync the `notebook` dependency group:

```bash
uv sync --group notebook
```

### HF_HOME

Model weights and dataset downloads (e.g. ARC-Easy via `datasets`) are cached under `HF_HOME`. Set once in your shell profile:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

(Already appended to `~/.bashrc` on this machine; open a new shell or `source ~/.bashrc` to pick it up.)

### VS Code + Claude Code in WSL

- Open the folder from inside WSL: `code .` (or "WSL: Connect to WSL" -> Open Folder `~/llm_interp`). This puts the extension host, terminals, and Python all inside WSL.
- Install the Claude Code extension in the WSL window ("Install in WSL: Ubuntu" on the extension page); the Claude Code CLI is installed separately (native build, no Node/npm needed — see `claude.ai/install.sh`).
- Run "Python: Select Interpreter" and pick `./.venv/bin/python` (also pre-filled via `.vscode/settings.json`).

## Running tests

```bash
uv run pytest
```

Unit tests run on a tiny, randomly-initialized Qwen3 config (CPU, no download). Integration tests marked `gpu` require the local GPU and will download `Qwen/Qwen3-0.6B`.

## Commands

The `llm-interp` console script (installed by `uv sync`) wraps five subcommands, run via `uv run llm-interp <command>`. Every subcommand takes `--config PATH` (defaults to `configs/qwen3_0p6b_local.yaml`).

- **`doctor`** — runs validation checks (a-f) against the configured model and prints a pass/fail table; exits non-zero on any failure. The first command to run on any machine/config.

  ```bash
  uv run llm-interp doctor
  ```

- **`estimate`** — projects storage bytes for a capture run from the config alone (fetches the model's `config.json`, never the weights). Run before any capture, especially before scaling up layers/tokens.

  ```bash
  uv run llm-interp estimate --num-traces 60 --tokens-per-trace 868
  ```

- **`generate`** — pulls `--num-pairs` ARC-Easy questions (default 30), runs the no-hint-then-hint paired generation for each, and writes the results to `<storage_root>/generated_traces.json`. Generation and capture are separate stages that only talk through this file.

  ```bash
  uv run llm-interp generate --num-pairs 30
  ```

- **`capture`** — loads `generated_traces.json` and teacher-force-replays every trace, writing the Zarr store, the four Parquet tables, and `manifest.json` under `storage_root`. Resumable: already-captured traces are skipped, and any data from a trace that crashed mid-write is truncated away on the next run.

  ```bash
  uv run llm-interp capture
  ```

- **`inspect`** — with no `--trace-id`, lists every captured trace (condition, parsed answer, parse status); with one, prints its metadata, spans, and decoded tokens.

  ```bash
  uv run llm-interp inspect
  uv run llm-interp inspect --trace-id Mercury_417466__hint
  ```

All of `generate`/`capture`/`inspect` accept `--storage-root PATH` to override the config's own `storage_root` (useful for running more than one experiment without editing the YAML).

## Notebook

`notebooks/explore_trace.ipynb` loads a captured trace (via the `Loader`), shows its tokens with think/response/hint/answer-value spans marked, plots per-layer residual norms, and recomputes one layer's keys as a sanity check against a fresh hooked forward pass. Run `generate` and `capture` first so there's something to load.
