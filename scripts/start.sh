#!/usr/bin/env bash
#
# start.sh — Start distill background services in a tmux session with caffeinate
#
# Usage:
#   ./scripts/start.sh                        # watchdog + dashboard
#   ./scripts/start.sh --monitor              # also start thermal monitor (no sudo needed)
#   ./scripts/start.sh --eval                 # start eval UI instead of dashboard
#   ./scripts/start.sh --backend=mlx          # pass backend to training launcher
#   ./scripts/start.sh --backend=unsloth      # pass backend to training launcher
#   ./scripts/start.sh --export=all           # pass export format to training launcher
#
# Services run in tmux session 'distill', each in a separate window, under caffeinate.
# Attach with: tmux attach -t distill
# Stop with:   ./scripts/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION="distill"

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

# Check dependencies
if ! command -v tmux &>/dev/null; then
  echo "Error: tmux not found. Install with: brew install tmux"
  exit 1
fi

# Kill any existing distill session and stale processes
echo "==> Cleaning up existing session and processes..."
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "[m]onitor_cpu_gpu_temp" 2>/dev/null || true
pkill -f "[t]raining_watchdog\.py" 2>/dev/null || true
pkill -f "[d]ashboard\.py"        2>/dev/null || true
pkill -f "[e]val_gradio\.py"      2>/dev/null || true
sleep 1

echo "==> Starting distill services in tmux session '$SESSION'..."

WATCHDOG_DIR="${DISTILL_OUTPUT_DIR:-$ROOT_DIR/distilled-minillm}"

# --- Window 0: Training watchdog ---
tmux new-session -d -s "$SESSION" -n "watchdog" -x 220 -y 50
tmux send-keys -t "$SESSION:watchdog" \
  "caffeinate -dims pixi run python $SCRIPT_DIR/training_watchdog.py $WATCHDOG_DIR --config $ROOT_DIR/configs/watchdog_rules.json 2>&1 | tee -a $ROOT_DIR/watchdog.log" Enter
echo "    watchdog       → $SESSION:watchdog"

# --- Window 1: Dashboard or Eval UI ---
tmux new-window -t "$SESSION" -n "dashboard"
if [ "$EVAL_UI" = true ]; then
  tmux send-keys -t "$SESSION:dashboard" \
    "caffeinate -dims pixi run python -m distill.eval_gradio 2>&1 | tee -a $ROOT_DIR/eval_gradio.log" Enter
  echo "    eval_gradio    → $SESSION:dashboard  (http://127.0.0.1:7860)"
else
  tmux send-keys -t "$SESSION:dashboard" \
    "caffeinate -dims pixi run python $SCRIPT_DIR/dashboard.py 2>&1 | tee -a $ROOT_DIR/dashboard.log" Enter
  echo "    dashboard      → $SESSION:dashboard  (http://127.0.0.1:7860)"
fi

# --- Window 2: Thermal monitor (optional) ---
if [ "$MONITOR" = true ]; then
  tmux new-window -t "$SESSION" -n "thermal"
  tmux send-keys -t "$SESSION:thermal" \
    "caffeinate -dims pixi run python $SCRIPT_DIR/monitor_cpu_gpu_temp.py --interval 3 --log $ROOT_DIR/thermal.log --fan-threshold 75 --fan-max-temp 90" Enter
  echo "    thermal        → $SESSION:thermal"
fi

tmux select-window -t "$SESSION:watchdog"

echo ""
echo "==> All services started."
echo "    Attach : tmux attach -t $SESSION"
echo "    Stop   : ./scripts/stop.sh"
