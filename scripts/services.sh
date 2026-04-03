#!/usr/bin/env bash
# services.sh — Manage distill background services
#
# Usage:
#   ./scripts/services.sh start [--monitor] [--eval] [--backend=mlx]
#   ./scripts/services.sh stop [--force]
#   ./scripts/services.sh monitor [interval_sec] [logfile]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CMD="${1:-}"

# ── start ────────────────────────────────────────────────────────────────────
cmd_start() {
  shift  # remove 'start'
  MONITOR=false; EVAL_UI=false; BACKEND=""; EXPORT=""
  for arg in "$@"; do
    case "$arg" in
      --monitor)   MONITOR=true ;;
      --eval)      EVAL_UI=true ;;
      --backend=*) BACKEND="${arg#*=}" ;;
      --export=*)  EXPORT="${arg#*=}" ;;
    esac
  done

  command -v tmux &>/dev/null || { echo "Error: tmux not found (brew install tmux)"; exit 1; }

  SESSION="${DISTILL_TMUX_SESSION:-distill}"
  echo "==> Cleaning up existing session and processes..."
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -f "[m]onitor_cpu_gpu_temp" 2>/dev/null || true
  pkill -f "[t]raining_watchdog"    2>/dev/null || true
  pkill -f "[d]ashboard\.py"        2>/dev/null || true
  pkill -f "distill\.ui\.dashboard" 2>/dev/null || true
  pkill -f "[e]val_gradio"          2>/dev/null || true
  sleep 1

  WATCHDOG_DIR="${DISTILL_OUTPUT_DIR:-$ROOT_DIR/distilled-minillm}"

  # Window 0: watchdog
  tmux new-session -d -s "$SESSION" -n "watchdog" -x 220 -y 50
  tmux send-keys -t "$SESSION:watchdog" \
    "caffeinate -dims pixi run python -m distill.training_watchdog $WATCHDOG_DIR --config $ROOT_DIR/configs/watchdog_rules.json 2>&1 | tee -a $ROOT_DIR/watchdog.log" Enter
  echo "    watchdog   -> $SESSION:watchdog"

  # Window 1: dashboard or eval UI
  tmux new-window -t "$SESSION" -n "dashboard"
  if [ "$EVAL_UI" = true ]; then
    tmux send-keys -t "$SESSION:dashboard" \
      "caffeinate -dims pixi run python -m distill.eval_gradio 2>&1 | tee -a $ROOT_DIR/eval_gradio.log" Enter
    echo "    eval_ui    -> $SESSION:dashboard  (http://127.0.0.1:7860)"
  else
    tmux send-keys -t "$SESSION:dashboard" \
      "caffeinate -dims pixi run python -m distill.ui.dashboard 2>&1 | tee -a $ROOT_DIR/dashboard.log" Enter
    echo "    dashboard  -> $SESSION:dashboard  (http://127.0.0.1:7860)"
  fi

  # Window 2: thermal monitor (optional)
  if [ "$MONITOR" = true ]; then
    tmux new-window -t "$SESSION" -n "thermal"
    tmux send-keys -t "$SESSION:thermal" \
      "caffeinate -dims pixi run python -m distill.monitoring.thermal --interval 3 --log $ROOT_DIR/thermal.log" Enter
    echo "    thermal    -> $SESSION:thermal"
  fi

  tmux select-window -t "$SESSION:watchdog"
  echo ""
  echo "==> Services running in tmux '$SESSION'"
  echo "    Attach : tmux attach -t $SESSION"
  echo "    Stop   : $(basename "$0") stop"
}

# ── stop ─────────────────────────────────────────────────────────────────────
cmd_stop() {
  shift  # remove 'stop'
  SIG="-TERM"
  for arg in "$@"; do [[ "$arg" == "--force" ]] && SIG="-KILL"; done

  _kill() {
    pkill $SIG -f "$1" 2>/dev/null && echo "    [killed] $2" || true
  }

  echo "==> Stopping distill services..."

  # Training processes
  _kill "[d]istill_minillm"         "distill_minillm"
  _kill "[d]istill_forward"         "distill_forward"
  _kill "[r]un_distillation_agent"  "run_distillation_agent"

  # Background daemons
  _kill "[t]raining_watchdog"       "training_watchdog"
  _kill "[m]onitor_cpu_gpu_temp"    "monitor_cpu_gpu_temp"
  _kill "distill\.monitoring"       "distill.monitoring"

  # Web UIs
  _kill "distill\.ui\.dashboard"    "dashboard"
  _kill "distill\.eval_gradio"      "eval_gradio"
  _kill "launch_ui"                 "launch_ui"

  # One-off utilities
  _kill "[p]lot_training"           "plot_training"
  _kill "[c]ache_models"            "cache_models"
  _kill "[c]ache_datasets"          "cache_datasets"

  # C++ binaries
  _kill "cpp/build/[w]atchdog"      "cpp watchdog"
  _kill "cpp/build/[d]istill"       "cpp distill"

  # PID file
  PID_FILE="$ROOT_DIR/.distill.pids"
  if [ -f "$PID_FILE" ]; then
    while IFS= read -r line; do
      pid=$(echo "$line" | awk '{print $1}')
      name=$(echo "$line" | awk '{$1=""; print $0}' | xargs)
      kill -0 "$pid" 2>/dev/null && kill $SIG "$pid" 2>/dev/null \
        && echo "    [killed] PID $pid ($name)" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi

  # tmux sessions — derived from DISTILL_TMUX_SESSION prefix so they match whatever was started
  _PFX="${DISTILL_TMUX_SESSION:-distill}"
  for sess in "$_PFX" "${_PFX}-phase2" "${_PFX}-export" "${_PFX}-download" "${_PFX}-thermal" "${_PFX}-prod"; do
    tmux has-session -t "$sess" 2>/dev/null \
      && tmux kill-session -t "$sess" && echo "    [killed] tmux: $sess" || true
  done

  echo "==> All distill processes stopped."
}

# ── monitor ──────────────────────────────────────────────────────────────────
cmd_monitor() {
  shift  # remove 'monitor'
  SESSION="${DISTILL_TMUX_SESSION:-distill}-thermal"
  INTERVAL="${1:-3}"
  LOGFILE="${2:-}"

  if [ -z "${TMUX:-}" ] && command -v tmux &>/dev/null; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 220 -y 50
    tmux send-keys -t "$SESSION" \
      "caffeinate -dims bash $(realpath "$0") monitor $INTERVAL $LOGFILE" Enter
    echo "==> Thermal monitor launched in tmux '$SESSION' (attach: tmux attach -t $SESSION)"
    exit 0
  fi

  command -v mactop &>/dev/null || { echo "Error: mactop not found (brew install mactop)"; exit 1; }

  HEADER="time                   CPU°C   GPU°C   SOC°C   CPU(W)  GPU(W)  Total(W)"
  DIVIDER="----------------------------------------------------------------------"
  echo "$HEADER"; echo "$DIVIDER"
  [ -n "$LOGFILE" ] && { echo "$HEADER"; echo "$DIVIDER"; } >> "$LOGFILE"

  while true; do
    JSON=$(mactop --headless --format json --count 1 | \
      python3 -c "import sys,json; d=json.load(sys.stdin)[0]['soc_metrics']; \
        print(d['cpu_temp'],d['gpu_temp'],d['soc_temp'],d['cpu_power'],d['gpu_power'],d['total_power'])")
    read -r CPU_T GPU_T SOC_T CPU_W GPU_W TOT_W <<< "$JSON"
    TS=$(date "+%Y-%m-%d %H:%M:%S")
    LINE=$(printf "%s   %5.1f   %5.1f   %5.1f   %5.2f   %5.2f   %7.2f" \
      "$TS" "$CPU_T" "$GPU_T" "$SOC_T" "$CPU_W" "$GPU_W" "$TOT_W")
    echo "$LINE"
    [ -n "$LOGFILE" ] && echo "$LINE" >> "$LOGFILE"
    sleep "$INTERVAL"
  done
}

# ── dispatch ─────────────────────────────────────────────────────────────────
case "$CMD" in
  start)   cmd_start "$@" ;;
  stop)    cmd_stop "$@" ;;
  monitor) cmd_monitor "$@" ;;
  *)
    echo "Usage: $(basename "$0") <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [--monitor] [--eval] [--backend=mlx]  start services in tmux"
    echo "  stop [--force]                               stop all distill processes"
    echo "  monitor [interval] [logfile]                 live thermal monitor"
    exit 1 ;;
esac
