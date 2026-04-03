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
SESSION="distill-phase2"

# If not inside tmux, relaunch in a new tmux session with caffeinate
if [ -z "${TMUX:-}" ]; then
  if ! command -v tmux &>/dev/null; then
    echo "Error: tmux not found. Install with: brew install tmux"
    exit 1
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  tmux send-keys -t "$SESSION" \
    "caffeinate -dims bash $(realpath "$0") $RUN; echo '==> Phase 2 done. Press Enter to close.'; read" Enter
  echo "==> Phase 2 launched in tmux session '$SESSION'"
  echo "    Attach with: tmux attach -t $SESSION"
  exit 0
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [phase2] $*"; }

run_variant() {
    local id="$1" cfg="$2"
    log "=== Starting run-$id (config: $cfg) ==="
    pixi run python -m distill.run_distillation_agent \
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
