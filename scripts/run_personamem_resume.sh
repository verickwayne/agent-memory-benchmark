#!/usr/bin/env bash
# Waits for cognee 128k (PID 4353) to finish, then runs remaining providers:
# mem0 (32k, 128k) and hindsight (32k, 128k)
# Logs appended to: logs/personamem_seq.log

set -uo pipefail
cd "$(dirname "$0")/.."

COGNEE_128K_PID=4353
MAX_RETRIES=3
WAIT_BETWEEN_RETRIES=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== resume script started ==="

run_one() {
    local memory="$1"
    local split="$2"
    log "  Running: dataset=personamem memory=$memory split=$split"
    uv run amb run \
        --dataset personamem \
        --split "$split" \
        --memory "$memory"
}

run_provider() {
    local memory="$1"
    shift
    local splits=("$@")
    log "========== Provider: $memory =========="
    local overall_failed=()
    for split in "${splits[@]}"; do
        log "--- Split: $split ---"
        local success=0
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
}

# cognee 32k done; cognee 128k was interrupted by disk-full, re-run it
run_provider cognee 128k
run_provider hindsight 32k 128k
run_provider mem0 32k 128k

log "=== All remaining runs done ==="
