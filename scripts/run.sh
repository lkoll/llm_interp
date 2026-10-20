#!/usr/bin/env bash
# Addendum B: run a long command inside a named tmux session, with logging to
# /workspace/runs/<run_id>/logs/, a heartbeat file while it's running, and a
# DONE marker + exit code at the end -- so a job started over SSH survives a
# dropped connection, and its outcome can be checked after the fact without
# re-attaching.
#
# Usage: ./run.sh [--stop-pod-when-done] <run_id> <command> [args...]

set -euo pipefail

STOP_POD_WHEN_DONE=0
if [[ "${1:-}" == "--stop-pod-when-done" ]]; then
  STOP_POD_WHEN_DONE=1
  shift
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--stop-pod-when-done] <run_id> <command> [args...]" >&2
  exit 1
fi

RUN_ID="$1"; shift
WORKSPACE="${WORKSPACE:-/workspace}"
RUN_DIR="$WORKSPACE/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/run_$STAMP.log"
HEARTBEAT_FILE="$RUN_DIR/heartbeat"
DONE_FILE="$RUN_DIR/DONE"
RUNNER_SCRIPT="$LOG_DIR/.runner_$STAMP.sh"
SESSION="run-$RUN_ID"

rm -f "$DONE_FILE"

# Built as its own script file, not one long inlined tmux command string --
# %q-quoting keeps this safe for arbitrary user commands/args, and construct-time
# bash control flow below (not quoted) decides whether the stop-pod block gets
# written in at all.
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -uo pipefail\n'   # no -e: a failing user command must still fall through to write DONE
  printf 'HEARTBEAT_FILE=%q\n' "$HEARTBEAT_FILE"
  printf 'DONE_FILE=%q\n' "$DONE_FILE"
  printf 'date -u +%%s > "$HEARTBEAT_FILE"\n'
  printf '(while true; do date -u +%%s > "$HEARTBEAT_FILE"; sleep 30; done) &\n'
  printf 'HEARTBEAT_PID=$!\n'
  printf 'trap '\''kill "$HEARTBEAT_PID" 2>/dev/null'\'' EXIT\n'
  printf '%q ' "$@"
  printf '\n'
  printf 'EXIT_CODE=$?\n'
  printf 'echo "$EXIT_CODE" > "$DONE_FILE"\n'
  if [[ "$STOP_POD_WHEN_DONE" -eq 1 ]]; then
    printf 'if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then\n'
    printf '  runpodctl stop pod "$RUNPOD_POD_ID"\n'
    printf 'else\n'
    printf '  echo "WARNING: --stop-pod-when-done requested but runpodctl or RUNPOD_POD_ID is unavailable -- skipping." >&2\n'
    printf 'fi\n'
  fi
  printf 'exit "$EXIT_CODE"\n'
} > "$RUNNER_SCRIPT"
chmod +x "$RUNNER_SCRIPT"

tmux new-session -d -s "$SESSION" "'$RUNNER_SCRIPT' 2>&1 | tee -a '$LOG_FILE'"

echo "Started tmux session '$SESSION' (tmux attach -t $SESSION to watch)."
echo "  log:       $LOG_FILE"
echo "  heartbeat: $HEARTBEAT_FILE"
echo "  done:      $DONE_FILE (exit code, written when the command finishes)"
