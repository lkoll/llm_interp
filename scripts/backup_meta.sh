#!/usr/bin/env bash
# Addendum B: back up the small, irreplaceable files from a run -- the four
# Parquet tables, the manifest, and the generate/capture handoff file -- kept
# separate from the (large, regenerable) Zarr activations. Default destination
# is meta/, committed to the repo, per the storage policy: "precious" data is
# small enough to just check in.
#
# Usage: ./backup_meta.sh <storage_root> [dest_dir]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <storage_root> [dest_dir]" >&2
  exit 1
fi

STORAGE_ROOT="$1"
RUN_NAME="$(basename "$STORAGE_ROOT")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${2:-$REPO_ROOT/meta/$RUN_NAME}"

mkdir -p "$DEST_DIR"

# Matches store/tables.py's TABLE_FILENAMES, capture.py's manifest.json, and
# cli.py's GENERATED_TRACES_FILENAME -- the activations.zarr/ directory next to
# these is deliberately excluded: it's regenerable from token_ids at the same
# model revision + dtype, not precious.
PRECIOUS_FILES=(
  "index.parquet"
  "traces.parquet"
  "spans.parquet"
  "annotations.parquet"
  "manifest.json"
  "generated_traces.json"
)

copied=0
missing=()
for f in "${PRECIOUS_FILES[@]}"; do
  if [[ -f "$STORAGE_ROOT/$f" ]]; then
    cp "$STORAGE_ROOT/$f" "$DEST_DIR/$f"
    copied=$((copied + 1))
  else
    missing+=("$f")
  fi
done

echo "Backed up $copied file(s) from $STORAGE_ROOT -> $DEST_DIR"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Not present (skipped): ${missing[*]}"
fi
echo "Note: manifest.json already embeds the full resolved config used for this run" \
     "(its config.model_dump() snapshot) -- copy the actual configs/*.yaml too if you" \
     "also want the original YAML file, not just the resolved values."
