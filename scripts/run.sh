#!/usr/bin/env bash
# run.sh — Distillation run launcher
#
# Usage:
#   ./scripts/run.sh production           # 5-trial autonomous search
#   ./scripts/run.sh golden               # golden pipeline config
#   ./scripts/run.sh phase2 [21|22|23]    # phase 2 runs (all or specific)
#   ./scripts/run.sh smoke                # quick smoke test (~5 min)
#   ./scripts/run.sh download [output_dir] # download Bartowski GGUF models
#   ./scripts/run.sh export [student_dir] [llama_cpp_dir] # export student to GGUF

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CMD="${1:-}"

# ── tmux helper ──────────────────────────────────────────────────────────────
_tmux_wrap() {
  local session="$1"; shift
  if [ -z "${TMUX:-}" ] && command -v tmux &>/dev/null; then
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" -x 220 -y 50
    tmux send-keys -t "$session" \
      "caffeinate -dims bash $(realpath "$0") $*; echo '==> Done. Press Enter.'; read" Enter
    echo "==> Launched in tmux session '$session'  (attach: tmux attach -t $session)"
    exit 0
  fi
}

# ── production ───────────────────────────────────────────────────────────────
cmd_production() {
  cd "$ROOT_DIR"
  SESSION="${DISTILL_TMUX_PREFIX:-distill}-prod"
  if [ -z "${TMUX:-}" ]; then
    if command -v tmux &>/dev/null; then
      if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
        echo "Kill: tmux kill-session -t $SESSION"
        exit 1
      fi
      echo "Launching in tmux '$SESSION'... (detach: Ctrl-B D)"
      exec tmux new-session -s "$SESSION" "$0" production  # re-entry has DISTILL_TMUX_PREFIX inherited
    fi
  fi

  echo "════════════════════════════════════════════════════════"
  echo "  Autonomous Production Distillation (5 trials)"
  echo "  Expected: 3.0–3.5 h  |  caffeinate active"
  echo "════════════════════════════════════════════════════════"
  echo "Starting in 3 seconds... (Ctrl+C to cancel)"; sleep 3

  caffeinate -i sleep 360000 &
  caffeinate -s \
  pixi run python -m distill.run_distillation_agent \
    --open --offline --n_trials 5 --curriculum --sft_epochs 1 --epochs 2 \
    --synthetic_data --n_synthetic 1500 --benchmarks --compare_teacher \
    --export gguf --log_experiment --watchdog

  echo ""; echo "Done! Next: python -m distill.experiment_log --show 5"
}

# ── golden ───────────────────────────────────────────────────────────────────
cmd_golden() {
  cd "$ROOT_DIR"
  mkdir -p runs
  exec .pixi/envs/default/bin/python \
    -m distill.run_distillation_agent --config configs/golden_pipeline.json "$@"
}

# ── phase2 ───────────────────────────────────────────────────────────────────
cmd_phase2() {
  local RUN="${2:-all}"
  _tmux_wrap "${DISTILL_TMUX_PREFIX:-distill}-phase2" phase2 "$RUN"
  cd "$ROOT_DIR"

  log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [phase2] $*"; }
  run_variant() {
    local id="$1" cfg="$2"
    log "=== Starting run-$id (config: $cfg) ==="
    pixi run python -m distill.run_distillation_agent \
      --config "$cfg" 2>&1 | tee "runs/phase2-run${id}.log"
    log "=== Finished run-$id ==="
  }

  mkdir -p runs
  [[ "$RUN" == "all" || "$RUN" == "21" ]] && run_variant 21 configs/phase2_run21.json
  [[ "$RUN" == "all" || "$RUN" == "22" ]] && run_variant 22 configs/phase2_run22.json
  [[ "$RUN" == "all" || "$RUN" == "23" ]] && run_variant 23 configs/phase2_run23.json
  log "Phase 2 complete."
}

# ── smoke ────────────────────────────────────────────────────────────────────
cmd_smoke() {
  cd "$ROOT_DIR"
  TEST_DIR="./smoke_test_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$TEST_DIR"
  echo "════════════════════════════════════════════════════════"
  echo "  Smoke Test — 50 samples, 1 epoch (~5 min)"
  echo "  Output: $TEST_DIR"
  echo "════════════════════════════════════════════════════════"

  pixi run python -m distill.distill_minillm \
    --open --offline --output_dir "$TEST_DIR" \
    --epochs 1 --max_samples 50 --eval_steps 2 --batch_size 8 --grad_acc 8 \
    2>&1 | tee "$TEST_DIR/distill.log"

  pixi run python -m distill.eval_quality "$TEST_DIR" \
    --student Qwen/Qwen2-0.5B-Instruct \
    --n_samples 10 --batch_size 4 --offline \
    2>&1 | tee "$TEST_DIR/quality.log"

  echo ""; echo "Results:"
  grep -q "Flash Attention 2 detected"    "$TEST_DIR/distill.log" && echo "  FA2: ENABLED"  || echo "  FA2: not installed"
  grep -q "torch.compile()"               "$TEST_DIR/distill.log" && echo "  compile: OK"   || true
  grep -q "Early stopping"                "$TEST_DIR/distill.log" && echo "  early-stop: OK" || true
  [ -f "$TEST_DIR/quality_metrics.json"  ] && echo "  quality metrics: OK" || echo "  quality metrics: MISSING"
  [ -f "$TEST_DIR/pytorch_model.bin" ] || [ -f "$TEST_DIR/model.safetensors" ] \
    && echo "" && echo "  SMOKE TEST PASSED" \
    || { echo "  SMOKE TEST FAILED — no model weights"; exit 1; }
}

# ── download ─────────────────────────────────────────────────────────────────
cmd_download() {
  _tmux_wrap "${DISTILL_TMUX_PREFIX:-distill}-download" download "${2:-}"
  OUTPUT_DIR="${2:-./gguf_models}"
  mkdir -p "$OUTPUT_DIR"
  BASE="https://huggingface.co"

  curl_dl() {
    local repo="$1" file="$2"
    local dest="$OUTPUT_DIR/$file"
    [[ -f "$dest" ]] && { echo "Exists: $dest"; return 0; }
    local url="$BASE/$repo/resolve/main/$file"
    echo "Downloading $file..."
    if [[ -n "${HF_TOKEN:-}" ]]; then
      curl -L -H "Authorization: Bearer $HF_TOKEN" -o "$dest" "$url"
    else
      curl -L -o "$dest" "$url"
    fi
  }

  curl_dl "bartowski/Llama-3.2-1B-Instruct-GGUF"     "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  curl_dl "bartowski/Dolphin3.0-Llama3.2-3B-GGUF"    "Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf"
  curl_dl "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  echo "Done. Models in $OUTPUT_DIR"
}

# ── export ───────────────────────────────────────────────────────────────────
cmd_export() {
  _tmux_wrap "${DISTILL_TMUX_PREFIX:-distill}-export" export "${2:-}" "${3:-}"
  STUDENT_DIR="${2:-./distilled-minillm}"
  LLAMA_CPP="${3:-}"
  OUTTYPE="${OUTTYPE:-f16}"

  if [[ -z "$LLAMA_CPP" ]]; then
    if   [[ -n "${LLAMA_CPP_ROOT:-}" && -d "$LLAMA_CPP_ROOT" ]]; then LLAMA_CPP="$LLAMA_CPP_ROOT"
    elif [[ -d "/Users/Shared/llama" ]];  then LLAMA_CPP="/Users/Shared/llama"
    elif [[ -d "./llama.cpp" ]];          then LLAMA_CPP="./llama.cpp"
    else                                       LLAMA_CPP="../llama.cpp"; fi
  fi

  [[ -d "$STUDENT_DIR" ]] || { echo "Error: student dir not found: $STUDENT_DIR"; exit 1; }
  CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
  [[ -f "$CONVERT" ]] || { echo "Error: convert_hf_to_gguf.py not found at $CONVERT"; exit 1; }

  STUDENT_ABS=$(cd "$STUDENT_DIR" && pwd)
  _LLAMA_MODELS_ROOT="${LLAMA_CPP_ROOT:-/Users/Shared/llama}"
  OUT_FILE="$_LLAMA_MODELS_ROOT/models/$(basename "$STUDENT_ABS")-${OUTTYPE}.gguf"
  mkdir -p "$(dirname "$OUT_FILE")"
  echo "Converting $STUDENT_ABS -> $OUT_FILE"
  python "$CONVERT" "$STUDENT_ABS" --outfile "$OUT_FILE" --outtype "$OUTTYPE"
  echo "Done. Run: $_LLAMA_MODELS_ROOT/llama-server -m $OUT_FILE"
}

# ── dispatch ─────────────────────────────────────────────────────────────────
case "$CMD" in
  production) cmd_production ;;
  golden)     cmd_golden "$@" ;;
  phase2)     cmd_phase2 "$@" ;;
  smoke)      cmd_smoke ;;
  download)   cmd_download "$@" ;;
  export)     cmd_export "$@" ;;
  *)
    echo "Usage: $(basename "$0") <command> [args]"
    echo ""
    echo "Commands:"
    echo "  production              5-trial autonomous distillation"
    echo "  golden                  golden pipeline config run"
    echo "  phase2 [21|22|23]       phase 2 config runs (default: all)"
    echo "  smoke                   quick smoke test (50 samples, ~5 min)"
    echo "  download [dir]          download Bartowski GGUF models"
    echo "  export [student] [llama] export student checkpoint to GGUF"
    exit 1 ;;
esac
