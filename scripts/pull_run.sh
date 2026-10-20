#!/usr/bin/env bash
# Addendum B: rsync a finished run from a RunPod pod to a local archive path
# over full SSH, verify it byte-for-byte with an independent sha256 pass (not
# just rsync's own transfer-time checksums), and optionally wipe the remote
# copy once verified.
#
# Usage:
#   ./pull_run.sh <run_id> <dest> --host <ip> [--port 22] [--user root] \
#       [--identity ~/.ssh/id_ed25519_runpod_wsl] [--remote-dir /workspace/runs/<run_id>] \
#       [--delete-remote-after-verify]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_id> <dest> --host <ip> [--port 22] [--user root] [--identity path] [--remote-dir path] [--delete-remote-after-verify]" >&2
  exit 1
fi

RUN_ID="$1"; shift
DEST="$1"; shift

HOST=""
PORT=22
SSH_USER="root"
IDENTITY="$HOME/.ssh/id_ed25519_runpod_wsl"
REMOTE_DIR=""
DELETE_REMOTE_AFTER_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --delete-remote-after-verify) DELETE_REMOTE_AFTER_VERIFY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "ERROR: --host is required (the pod's SSH IP, from the RunPod console)." >&2
  exit 1
fi
REMOTE_DIR="${REMOTE_DIR:-/workspace/runs/$RUN_ID}"

SSH_OPTS=(-p "$PORT" -i "$IDENTITY" -o StrictHostKeyChecking=accept-new)
LOCAL_RUN_DIR="$DEST/$RUN_ID"
mkdir -p "$LOCAL_RUN_DIR"

log() { printf '[pull_run] %s\n' "$*"; }

log "rsync: $SSH_USER@$HOST:$REMOTE_DIR/ -> $LOCAL_RUN_DIR/"
rsync -avz --checksum \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SSH_USER@$HOST:$REMOTE_DIR/" "$LOCAL_RUN_DIR/"

log "Verifying with an independent sha256 pass (not just rsync's own transfer checksums)..."
REMOTE_MANIFEST="$(mktemp)"
LOCAL_MANIFEST="$(mktemp)"
trap 'rm -f "$REMOTE_MANIFEST" "$LOCAL_MANIFEST"' EXIT

ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" \
  "cd '$REMOTE_DIR' && find . -type f -exec sha256sum {} \;" | sort > "$REMOTE_MANIFEST"
(cd "$LOCAL_RUN_DIR" && find . -type f -exec sha256sum {} \;) | sort > "$LOCAL_MANIFEST"

if ! diff -u "$REMOTE_MANIFEST" "$LOCAL_MANIFEST"; then
  log "ERROR: checksum mismatch between remote and local copy -- NOT deleting the remote copy. See the diff above."
  exit 1
fi
log "Checksums match: $(wc -l < "$LOCAL_MANIFEST") file(s) verified."

if [[ "$DELETE_REMOTE_AFTER_VERIFY" -eq 1 ]]; then
  log "Deleting verified remote copy at $REMOTE_DIR..."
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "rm -rf '$REMOTE_DIR'"
fi

log "Done. Archive: $LOCAL_RUN_DIR"
