#!/usr/bin/env bash
# gradio.sh — Launch the Universal Model Evaluator UI
#
# Usage:
#   ./scripts/gradio.sh                            # auto-detect model, port 7860
#   ./scripts/gradio.sh --model_path ./my-model    # specific model
#   ./scripts/gradio.sh --backend gguf --port 7861 # force backend + port
#   ./scripts/gradio.sh --no-tmux                  # skip tmux, run in foreground

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${DISTILL_TMUX_PREFIX:-distill}-gradio"
PORT=7860
NO_TMUX=false

# Extract --port and --no-tmux before forwarding remaining args
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; EXTRA_ARGS+=("--port" "$2"); shift 2 ;;
    --port=*) PORT="${1#*=}"; EXTRA_ARGS+=("$1"); shift ;;
    --no-tmux) NO_TMUX=true; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

cd "$ROOT_DIR"

# Kill any existing gradio session on the same port
pkill -f "distill\.eval\.gradio_ui.*--port $PORT" 2>/dev/null || true
tmux kill-session -t "$SESSION" 2>/dev/null || true

RUN_CMD="KMP_DUPLICATE_LIB_OK=TRUE caffeinate -i .pixi/envs/default/bin/python \
  -m distill.eval.gradio_ui ${EXTRA_ARGS[*]+"${EXTRA_ARGS[@]}"} 2>&1 | tee -a $ROOT_DIR/gradio.log"

if [ "$NO_TMUX" = true ] || [ -n "${TMUX:-}" ] || ! command -v tmux &>/dev/null; then
  echo "==> Starting Gradio UI on http://127.0.0.1:${PORT}"
  eval "$RUN_CMD"
else
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  tmux send-keys -t "$SESSION" "$RUN_CMD; echo '==> Exited. Press Enter.'; read" Enter
  echo "==> Gradio UI launching in tmux session '$SESSION'"
  echo "    URL    : http://127.0.0.1:${PORT}"
  echo "    Attach : tmux attach -t $SESSION"
  echo "    Stop   : tmux kill-session -t $SESSION"
fi
