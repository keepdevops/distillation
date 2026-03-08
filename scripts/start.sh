#!/usr/bin/env bash
#
# start.sh — Start distill background services
#
# Usage:
#   ./scripts/start.sh                        # watchdog + dashboard
#   ./scripts/start.sh --monitor              # also start thermal monitor (no sudo needed)
#   ./scripts/start.sh --eval                 # start eval UI instead of dashboard
#   ./scripts/start.sh --backend=mlx          # pass backend to training launcher
#   ./scripts/start.sh --backend=unsloth      # pass backend to training launcher
#   ./scripts/start.sh --export=all           # pass export format to training launcher
#
# Services started:
#   - training_watchdog.py  (background daemon, PID logged)
#   - dashboard.py          (Gradio UI at http://127.0.0.1:7860)
#   - [optional] monitor_cpu_gpu_temp.py  (uses mactop, no sudo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/.distill.pids"

# Parse flags
MONITOR=false
EVAL_UI=false
BACKEND=""
EXPORT=""
for arg in "$@"; do
  case "$arg" in
    --monitor)   MONITOR=true ;;
    --eval)      EVAL_UI=true ;;
    --backend=*) BACKEND="${arg#*=}" ;;
    --backend)   ;;  # use --backend=mlx form
    --export=*)  EXPORT="${arg#*=}" ;;
  esac
done

# Kill any already-running instances of managed services to prevent duplicates
pkill -f "[m]onitor_cpu_gpu_temp" 2>/dev/null || true
pkill -f "[t]raining_watchdog\.py" 2>/dev/null || true
pkill -f "[d]ashboard\.py"        2>/dev/null || true
pkill -f "[e]val_gradio\.py"      2>/dev/null || true
sleep 1

# Clear old PID file
> "$PID_FILE"

echo "==> Starting distill services..."

# --- Training watchdog ---
echo -n "    watchdog       ... "
WATCHDOG_DIR="${DISTILL_OUTPUT_DIR:-$ROOT_DIR/distilled-minillm}"
python "$SCRIPT_DIR/training_watchdog.py" "$WATCHDOG_DIR" \
  --config "$ROOT_DIR/configs/watchdog_rules.json" \
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

# --- Thermal monitor (optional, no sudo needed — uses mactop) ---
if [ "$MONITOR" = true ]; then
  echo -n "    thermal monitor... "
  python "$SCRIPT_DIR/monitor_cpu_gpu_temp.py" \
    --interval 3 --log "$ROOT_DIR/thermal.log" \
    --fan-threshold 75 --fan-max-temp 90 \
    >> /dev/null 2>&1 &
  MON_PID=$!
  echo "$MON_PID  monitor_cpu_gpu_temp.py" >> "$PID_FILE"
  echo "PID $MON_PID"
fi

echo ""
echo "==> Services running. PIDs saved to $PID_FILE"
echo "    Stop with: ./scripts/stop.sh"
