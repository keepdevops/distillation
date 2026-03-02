#!/usr/bin/env bash
#
# start.sh — Start distill background services
#
# Usage:
#   ./scripts/start.sh              # watchdog + dashboard
#   ./scripts/start.sh --monitor    # also start thermal monitor (requires sudo)
#   ./scripts/start.sh --eval       # start eval UI instead of dashboard
#
# Services started:
#   - training_watchdog.py  (background daemon, PID logged)
#   - dashboard.py          (Gradio UI at http://127.0.0.1:7860)
#   - [optional] monitor_cpu_gpu_temp.sh  (requires sudo)
#   - [optional] eval_gradio.py           (replaces dashboard)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/.distill.pids"

# Parse flags
MONITOR=false
EVAL_UI=false
for arg in "$@"; do
  case "$arg" in
    --monitor) MONITOR=true ;;
    --eval)    EVAL_UI=true ;;
  esac
done

# Clear old PID file
> "$PID_FILE"

echo "==> Starting distill services..."

# --- Training watchdog ---
echo -n "    watchdog       ... "
WATCHDOG_DIR="${DISTILL_OUTPUT_DIR:-$ROOT_DIR/distilled-minillm}"
python "$SCRIPT_DIR/training_watchdog.py" "$WATCHDOG_DIR" \
  >> "$ROOT_DIR/watchdog.log" 2>&1 &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID  training_watchdog.py" >> "$PID_FILE"
echo "PID $WATCHDOG_PID"

# --- Dashboard or Eval UI ---
if [ "$EVAL_UI" = true ]; then
  echo -n "    eval_gradio    ... "
  python "$SCRIPT_DIR/eval_gradio.py" \
    >> "$ROOT_DIR/eval_gradio.log" 2>&1 &
  UI_PID=$!
  echo "$UI_PID  eval_gradio.py" >> "$PID_FILE"
  echo "PID $UI_PID  → http://127.0.0.1:7860"
else
  echo -n "    dashboard      ... "
  python "$SCRIPT_DIR/dashboard.py" \
    >> "$ROOT_DIR/dashboard.log" 2>&1 &
  UI_PID=$!
  echo "$UI_PID  dashboard.py" >> "$PID_FILE"
  echo "PID $UI_PID  → http://127.0.0.1:7860"
fi

# --- Thermal monitor (optional, requires sudo) ---
if [ "$MONITOR" = true ]; then
  echo -n "    thermal monitor... "
  sudo "$SCRIPT_DIR/monitor_cpu_gpu_temp.sh" \
    >> "$ROOT_DIR/thermal.log" 2>&1 &
  MON_PID=$!
  echo "$MON_PID  monitor_cpu_gpu_temp.sh" >> "$PID_FILE"
  echo "PID $MON_PID"
fi

echo ""
echo "==> Services running. PIDs saved to $PID_FILE"
echo "    Stop with: ./scripts/stop.sh"
