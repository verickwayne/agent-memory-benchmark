#!/bin/bash
# Resilient restart loop for longmemeval hindsight run.
# Keeps retrying with --skip-ingested until all 500 units are done.
set -euo pipefail
cd "$(dirname "$0")/.."

get_progress() {
  python3 -c "
import json, sys
try:
    with open('outputs/longmemeval/hindsight/rag/s.json') as f:
        data = json.load(f)
    print(len(data['results']))
except:
    print(0)
"
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  progress=$(get_progress)
  echo "[attempt $attempt] Progress: $progress/500 — starting run..."

  if [ "$progress" -ge 500 ]; then
    echo "All 500 units complete!"
    break
  fi

  uv run amb run --dataset longmemeval --split s --memory hindsight --skip-ingested || true

  new_progress=$(get_progress)
  echo "[attempt $attempt] After run: $new_progress/500"

  if [ "$new_progress" -ge 500 ]; then
    echo "All 500 units complete!"
    break
  fi

  if [ "$new_progress" -eq "$progress" ]; then
    # No progress made — wait longer before retrying (rate limits)
    echo "No progress made, waiting 60s before retry..."
    sleep 60
  else
    echo "Progress made ($progress → $new_progress), retrying in 5s..."
    sleep 5
  fi
done

echo "Done. Final progress: $(get_progress)/500"
