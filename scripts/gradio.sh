#!/usr/bin/env bash
#
# gradio.sh — Launch the Gradio eval UI (distill.eval.gradio_ui)
#
# Usage:
#   ./scripts/gradio.sh            # foreground, logs to stdout
#   ./scripts/gradio.sh --bg       # background via tmux window in 'distill' session
#   ./scripts/gradio.sh --port=7861  # override default port (default: 7860)
#
# Attach (if --bg):  tmux attach -t distill
# Stop (if --bg):    tmux kill-window -t distill:gradio

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION="distill"
PORT=7860
BG=false

for arg in "$@"; do
  case "$arg" in
    --bg)       BG=true ;;
    --port=*)   PORT="${arg#*=}" ;;
  esac
done

CMD="GRADIO_SERVER_PORT=$PORT pixi run python -m distill.eval_gradio"
LOG="$ROOT_DIR/eval_gradio.log"

if [ "$BG" = true ]; then
  # Attach to existing distill session or create a new one
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "gradio" -x 220 -y 50
  else
    tmux new-window -t "$SESSION" -n "gradio" 2>/dev/null || \
      tmux select-window -t "$SESSION:gradio"
  fi
  tmux send-keys -t "$SESSION:gradio" \
    "caffeinate -dims $CMD 2>&1 | tee -a $LOG" Enter
  echo "==> Gradio UI started in tmux ($SESSION:gradio)"
  echo "    URL  : http://127.0.0.1:$PORT"
  echo "    Log  : $LOG"
  echo "    Stop : tmux kill-window -t $SESSION:gradio"
else
  echo "==> Launching Gradio UI on http://127.0.0.1:$PORT"
  cd "$ROOT_DIR"
  exec env GRADIO_SERVER_PORT="$PORT" pixi run python -m distill.eval_gradio
fi
