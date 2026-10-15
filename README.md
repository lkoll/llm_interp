# llm_interp

Activation capture pipeline for interpretability research on LLM reasoning (Phase 0: Qwen3-0.6B, local GPU). See `project_outline.md` for the full brief, `task_list.md` for progress, and `CLAUDE.md` / `DECISIONS.md` for agreed conventions and design decisions.

## Setup (WSL2 Ubuntu)

Prerequisites: WSL2 Ubuntu with GPU passthrough (`nvidia-smi` should show your GPU), repo checked out on the WSL filesystem (not `/mnt/c`), [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv sync
```

This creates `.venv` using a uv-managed Python 3.12 (pinned via `.python-version`), independent of the system Python.

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

CLI commands (`doctor`, `estimate`, `generate`, `capture`, `inspect`) land as the corresponding build steps are completed — see `task_list.md`.
