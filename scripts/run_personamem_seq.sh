#!/usr/bin/env bash
# Sequentially run personamem benchmark for mem0, cognee, hindsight (local)
# across all splits: 32k, 128k, 1M
# Retries on failure, up to MAX_RETRIES per run.
# Logs to: logs/personamem_seq.log

set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDERS=(mem0 cognee hindsight)
SPLITS=(32k 128k)
MAX_RETRIES=3
WAIT_BETWEEN_RETRIES=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local memory="$1"
    local split="$2"
    log "  Running: dataset=personamem memory=$memory split=$split"
    uv run amb run \
        --dataset personamem \
        --split "$split" \
        --memory "$memory"
}

log "=== personamem sequential run started ==="
log "Providers: ${PROVIDERS[*]}"
log "Splits: ${SPLITS[*]}"

overall_failed=()

for memory in "${PROVIDERS[@]}"; do
    log "========== Provider: $memory =========="
    for split in "${SPLITS[@]}"; do
        log "--- Split: $split ---"
        success=0
        for attempt in $(seq 1 $MAX_RETRIES); do
            log "  attempt $attempt/$MAX_RETRIES"
            if run_one "$memory" "$split"; then
                log "  SUCCESS: memory=$memory split=$split"
                success=1
                break
            else
                log "  FAILED (attempt $attempt) — waiting ${WAIT_BETWEEN_RETRIES}s before retry"
                sleep $WAIT_BETWEEN_RETRIES
            fi
        done
        if [[ "$success" -eq 0 ]]; then
            log "  GIVING UP on memory=$memory split=$split after $MAX_RETRIES attempts"
            overall_failed+=("$memory/$split")
        fi
    done
    log "========== Done with provider: $memory =========="
done

log "=== All done ==="
if [[ ${#overall_failed[@]} -gt 0 ]]; then
    log "FAILED runs:"
    for f in "${overall_failed[@]}"; do
        log "  - $f"
    done
    exit 1
else
    log "All runs completed successfully!"
fi
