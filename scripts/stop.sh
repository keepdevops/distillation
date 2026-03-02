#!/usr/bin/env bash
#
# stop.sh — Stop all distill app processes
#
# Usage:
#   ./scripts/stop.sh          # graceful SIGTERM then SIGKILL
#   ./scripts/stop.sh --force  # immediate SIGKILL

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/.distill.pids"

FORCE=false
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE=true
done

SIG="-TERM"
$FORCE && SIG="-KILL"

pkill_safe() {
  local pattern="$1"
  local label="$2"
  if pkill $SIG -f "$pattern" 2>/dev/null; then
    echo "    [killed] $label"
  fi
}

echo "==> Stopping distill services..."

# --- Training scripts (batch processes that may be running) ---
pkill_safe "[d]istill_minillm\.py"         "distill_minillm.py"
pkill_safe "[d]istill_forward\.py"         "distill_forward.py"
pkill_safe "[r]un_distillation_agent\.py"  "run_distillation_agent.py"

# --- Background daemons ---
pkill_safe "[t]raining_watchdog\.py"       "training_watchdog.py"
pkill_safe "[m]onitor_cpu_gpu_temp\.sh"    "monitor_cpu_gpu_temp.sh"
pkill_safe "[m]onitor_cpu_gpu_temp\.py"    "monitor_cpu_gpu_temp.py"

# --- Web UIs ---
pkill_safe "[d]ashboard\.py"              "dashboard.py"
pkill_safe "[e]val_gradio\.py"            "eval_gradio.py"

# --- One-off utilities that may be lingering ---
pkill_safe "[p]lot_training\.py"          "plot_training.py"
pkill_safe "[c]ache_models\.py"           "cache_models.py"
pkill_safe "[c]ache_datasets\.py"         "cache_datasets.py"

# --- powermetrics spawned by thermal monitor (requires sudo) ---
if pgrep -x powermetrics &>/dev/null; then
  echo -n "    [killing] powermetrics (requires sudo)... "
  sudo pkill $SIG -x powermetrics 2>/dev/null && echo "done" || echo "failed (check sudo)"
fi

# --- C++ binaries ---
pkill_safe "cpp/build/[w]atchdog"         "cpp watchdog"
pkill_safe "cpp/build/[d]istill"          "cpp distill"

# --- PID file cleanup ---
if [ -f "$PID_FILE" ]; then
  echo ""
  echo "    Stale PIDs from $PID_FILE:"
  while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $1}')
    name=$(echo "$line" | awk '{$1=""; print $0}' | xargs)
    if kill -0 "$pid" 2>/dev/null; then
      kill $SIG "$pid" 2>/dev/null && echo "      [killed] PID $pid ($name)"
    fi
  done < "$PID_FILE"
  rm -f "$PID_FILE"
fi

echo ""
echo "==> All distill processes stopped."
