#!/usr/bin/env bash
# Run evaluation benchmark when at least one clip exists under evaluation/clips/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

shopt -s nullglob
clips=(evaluation/clips/*.mp4)
if [[ ${#clips[@]} -eq 0 ]]; then
  echo "No evaluation/clips/*.mp4 found."
  echo "Add videos (see evaluation/manifest.csv paths) or run without this script:"
  echo "  python scripts/benchmark_pipeline.py --manifest evaluation/manifest.csv --out evaluation/results.jsonl"
  exit 0
fi

python scripts/benchmark_pipeline.py \
  --manifest evaluation/manifest.csv \
  --out evaluation/results.jsonl \
  --backend mediapipe

echo "Wrote evaluation/results.jsonl (${#clips[@]} clip(s) present; manifest may list more)."
