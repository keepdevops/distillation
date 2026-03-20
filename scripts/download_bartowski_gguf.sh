#!/usr/bin/env bash
#
# Download Bartowski GGUF models via curl (US origin).
# Bartowski repos are HuggingFace US region; resolve URLs route to cdn-lfs-us-1.
# Quantized GGUF for llama.cpp / LM Studio; not used for distillation.
#
# Usage:
#   ./download_bartowski_gguf.sh [output_dir]
#   HF_TOKEN=your_token ./download_bartowski_gguf.sh   # gated models (Llama 3.2)
#
# Models: https://huggingface.co/bartowski

set -e

SESSION="distill-download"

# If not inside tmux, relaunch in a new tmux session with caffeinate
if [ -z "${TMUX:-}" ]; then
  if ! command -v tmux &>/dev/null; then
    echo "Error: tmux not found. Install with: brew install tmux"
    exit 1
  fi
  ARGS="$*"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  tmux send-keys -t "$SESSION" \
    "caffeinate -dims bash $(realpath "$0") $ARGS; echo '==> Download done. Press Enter to close.'; read" Enter
  echo "==> Download launched in tmux session '$SESSION'"
  echo "    Attach with: tmux attach -t $SESSION"
  exit 0
fi

OUTPUT_DIR="${1:-./gguf_models}"
mkdir -p "$OUTPUT_DIR"

# HuggingFace resolve URL (US region for Bartowski; redirects to cdn-lfs-us-1)
BASE="https://huggingface.co"
REPO_1B="bartowski/Llama-3.2-1B-Instruct-GGUF"
REPO_3B="bartowski/Dolphin3.0-Llama3.2-3B-GGUF"
REPO_8B="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"

# Default: Q4_K_M (balanced quality/size)
FILE_1B="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
FILE_3B="Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf"
FILE_8B="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

curl_download() {
  local repo="$1"
  local file="$2"
  local dest="$OUTPUT_DIR/$file"
  local url="$BASE/$repo/resolve/main/$file"
  if [[ -f "$dest" ]]; then
    echo "Exists: $dest"
    return 0
  fi
  echo "Downloading $url (US origin)..."
  if [[ -n "$HF_TOKEN" ]]; then
    curl -L -H "Authorization: Bearer $HF_TOKEN" -o "$dest" "$url"
  else
    curl -L -o "$dest" "$url"
  fi
}

# Curl from Bartowski (US origin)
echo "Using curl (Bartowski repos are US region)"
curl_download "$REPO_1B" "$FILE_1B"
curl_download "$REPO_3B" "$FILE_3B"
curl_download "$REPO_8B" "$FILE_8B"
echo "Done. Models in $OUTPUT_DIR"
