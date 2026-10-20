#!/usr/bin/env bash
# Addendum B: idempotent setup for a RunPod pod. Run this right after connecting
# to a fresh pod, and again after any restart -- the container disk (and
# anything outside the network volume) is wiped then, so venvs get rebuilt from
# the uv cache on the volume each time.
#
# Env vars this script reads, set as pod env vars and never written to any file:
#   HF_TOKEN     - Hugging Face token, read natively by huggingface_hub for
#                  gated/private model downloads.
#   GITHUB_TOKEN - a GitHub PAT with read access to the private repo. Used only
#                  as a transient header on the clone/pull command, never
#                  embedded in a URL or written into .git/config.
#
# Usage: ./cloud_setup.sh [--config path/to/config.yaml] [--repo-url url]

set -euo pipefail

CONFIG_PATH="configs/qwen3_8b_cloud.yaml"
REPO_URL="${REPO_URL:-https://github.com/lkoll/llm_interp.git}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/llm_interp}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.cache/uv}"

log() { printf '[cloud_setup] %s\n' "$*"; }

log "Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
else
  log "WARNING: nvidia-smi not found -- no GPU visible on this pod. Continuing; the final 'doctor' run will catch this."
fi

log "Persisting HF_HOME/UV_CACHE_DIR to ~/.bashrc (idempotent)..."
persist_env() {
  local line="export $1=\"$2\""
  touch ~/.bashrc
  grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}
persist_env HF_HOME "$HF_HOME"
persist_env UV_CACHE_DIR "$UV_CACHE_DIR"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

log "Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [[ -d "$REPO_DIR/.git" ]]; then
  log "Repo already present at $REPO_DIR -- pulling..."
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    # GitHub's git-over-HTTPS endpoint (as opposed to its REST API) rejects a
    # Bearer header for a PAT with a flat 401; it wants HTTP Basic auth with
    # the token as the password. Confirmed against the live repo.
    AUTH_HEADER="Authorization: Basic $(printf '%s' "x-access-token:${GITHUB_TOKEN}" | base64 -w0)"
    git -C "$REPO_DIR" -c "http.extraHeader=${AUTH_HEADER}" pull
  else
    git -C "$REPO_DIR" pull
  fi
else
  log "Cloning $REPO_URL into $REPO_DIR..."
  mkdir -p "$(dirname "$REPO_DIR")"
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    AUTH_HEADER="Authorization: Basic $(printf '%s' "x-access-token:${GITHUB_TOKEN}" | base64 -w0)"
    git -c "http.extraHeader=${AUTH_HEADER}" clone "$REPO_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

cd "$REPO_DIR"

log "Syncing the main environment (uv sync -- torch's own wheels bundle a matching CUDA build)..."
uv sync

log "Building the separate vLLM environment..."
# Deliberately NOT added to pyproject.toml: task_list.md's Phase 1 spec calls for
# vLLM generation to live in "its own environment (own torch pin)", isolated
# from the main project's pinned deps. Left unpinned here until the real vLLM
# generator (generate/vllm.py is a documented stub today) is actually built --
# pin it then, not speculatively now.
if [[ ! -d "$WORKSPACE/.venv-vllm" ]]; then
  uv venv "$WORKSPACE/.venv-vllm"
fi
uv pip install --python "$WORKSPACE/.venv-vllm/bin/python" vllm

log "Deriving the storage root from $CONFIG_PATH's run_name (lands at $WORKSPACE/runs/<run_name>, per Addendum B's volume layout)..."
RUN_NAME="$(uv run python -c "
from llm_interp.config import load_config
print(load_config('$CONFIG_PATH').run_name)
")"
STORAGE_ROOT="$WORKSPACE/runs/$RUN_NAME"
mkdir -p "$STORAGE_ROOT"
log "storage_root = $STORAGE_ROOT"

log "Pre-downloading model weights into HF_HOME (HF_TOKEN read from env if set)..."
uv run python -c "
from llm_interp.config import load_config
from huggingface_hub import snapshot_download
config = load_config('$CONFIG_PATH')
path = snapshot_download(repo_id=config.model.id, revision=config.model.revision)
print(f'Downloaded {config.model.id}@{config.model.revision} -> {path}')
"

log "Running doctor with the cloud config..."
uv run llm-interp doctor --config "$CONFIG_PATH" --storage-root "$STORAGE_ROOT"
