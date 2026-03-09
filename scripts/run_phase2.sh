#!/usr/bin/env bash
# Phase 2 — Data Quality & Scale-Up launcher
# Runs run-2.1, run-2.2, run-2.3 sequentially.
# Usage:
#   bash scripts/run_phase2.sh           # all three runs
#   bash scripts/run_phase2.sh 21        # only run-2.1
#   bash scripts/run_phase2.sh 22        # only run-2.2
#   bash scripts/run_phase2.sh 23        # only run-2.3

set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-all}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [phase2] $*"; }

run_variant() {
    local id="$1" cfg="$2"
    log "=== Starting run-$id (config: $cfg) ==="
    python scripts/run_distillation_agent.py \
        --config "$cfg" \
        2>&1 | tee "runs/phase2-run${id}.log"
    log "=== Finished run-$id ==="
}

mkdir -p runs

if [[ "$RUN" == "all" || "$RUN" == "21" ]]; then
    run_variant 21 configs/phase2_run21.json
fi

if [[ "$RUN" == "all" || "$RUN" == "22" ]]; then
    run_variant 22 configs/phase2_run22.json
fi

if [[ "$RUN" == "all" || "$RUN" == "23" ]]; then
    run_variant 23 configs/phase2_run23.json
fi

log "Phase 2 complete. Check experiment_log.jsonl for results."
