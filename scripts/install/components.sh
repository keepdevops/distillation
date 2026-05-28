#!/usr/bin/env bash
# install/components.sh — Component installers (sourced by install.sh)
# Requires: MODEL_PATH, LLAMA_CPP_ROOT, THRESHOLD, INTERVAL, REPO_ROOT
# Requires: color helpers (ok, warn, error, info, section) from install.sh

install_shared_models() {
  section "Shared Model Storage"

  if [[ ! -d "$MODEL_PATH" ]]; then
    mkdir -p "$MODEL_PATH"
    chmod 755 "$MODEL_PATH"
  fi
  ok "Model path   : $MODEL_PATH"
  ls -lh "$MODEL_PATH" 2>/dev/null | tail -n +2 | awk '{print "    " $0}' || true

  local LLAMA_MODELS_DIR="$LLAMA_CPP_ROOT/models"
  if [[ -d "$LLAMA_CPP_ROOT" ]]; then
    if [[ ! -d "$LLAMA_MODELS_DIR" ]]; then
      mkdir -p "$LLAMA_MODELS_DIR"
      chmod 755 "$LLAMA_MODELS_DIR"
    fi
    ok "llama models : $LLAMA_MODELS_DIR"
    ls -lh "$LLAMA_MODELS_DIR" 2>/dev/null | tail -n +2 | awk '{print "    " $0}' || true
  else
    warn "llama.cpp not found at $LLAMA_CPP_ROOT — skipping GGUF model dir"
  fi

  _sync_zshrc
}

_sync_zshrc() {
  local ZSHRC="$HOME/.zshrc"
  [[ -f "$ZSHRC" ]] || return 0

  if grep -q "MODEL_PATH=" "$ZSHRC"; then
    sed -i '' "s|export MODEL_PATH=.*|export MODEL_PATH=$MODEL_PATH|" "$ZSHRC"
    ok "Updated MODEL_PATH in ~/.zshrc → $MODEL_PATH"
  else
    printf '\n# Distilled model storage\nexport MODEL_PATH=%s\n' "$MODEL_PATH" >> "$ZSHRC"
    ok "Added MODEL_PATH to ~/.zshrc"
    info "Reload: source ~/.zshrc"
  fi

  if [[ -d "$LLAMA_CPP_ROOT" ]]; then
    if grep -q "LLAMA_CPP_ROOT=" "$ZSHRC"; then
      sed -i '' "s|export LLAMA_CPP_ROOT=.*|export LLAMA_CPP_ROOT=$LLAMA_CPP_ROOT|" "$ZSHRC"
      ok "Updated LLAMA_CPP_ROOT in ~/.zshrc → $LLAMA_CPP_ROOT"
    else
      printf 'export LLAMA_CPP_ROOT=%s\n' "$LLAMA_CPP_ROOT" >> "$ZSHRC"
      ok "Added LLAMA_CPP_ROOT to ~/.zshrc"
      info "Reload: source ~/.zshrc"
    fi
  fi
}

install_thermal_agent() {
  section "Thermal Monitoring Agent"

  local LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
  local PLIST_PATH="$LAUNCH_AGENTS/com.distillation.thermal_agent.plist"
  local PYTHON_PATH
  PYTHON_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
  PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"

  info "Python: $PYTHON_PATH  Threshold: ${THRESHOLD}°C  Interval: ${INTERVAL}s"
  mkdir -p "$LAUNCH_AGENTS"

  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.distillation.thermal_agent</string>
  <key>ProgramArguments</key><array>
    <string>$PYTHON_PATH</string>
    <string>-m</string><string>distill.orchestration.agent</string>
    <string>--watch</string><string>$REPO_ROOT</string>
    <string>--threshold</string><string>$THRESHOLD</string>
    <string>--interval</string><string>$INTERVAL</string>
    <string>--log</string><string>$REPO_ROOT/thermal_agent.jsonl</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$REPO_ROOT/thermal_agent.stdout.log</string>
  <key>StandardErrorPath</key><string>$REPO_ROOT/thermal_agent.stderr.log</string>
  <key>WorkingDirectory</key><string>$REPO_ROOT</string>
  <key>Nice</key><integer>10</integer>
</dict></plist>
EOF

  launchctl list | grep -q com.distillation.thermal_agent \
    && launchctl unload "$PLIST_PATH" 2>/dev/null || true
  launchctl load "$PLIST_PATH"
  sleep 2

  if launchctl list | grep -q com.distillation.thermal_agent; then
    ok "Thermal agent installed and running."
  else
    error "Agent not running. Check: tail -f $REPO_ROOT/thermal_agent.stderr.log"
    exit 1
  fi
  info "Stop:  launchctl unload $PLIST_PATH"
  info "Start: launchctl load $PLIST_PATH"
}

install_vllm() {
  section "vLLM Install"

  if [[ "${OS:-}" == "Darwin" ]]; then
    warn "vLLM does not officially support macOS. Use gguf or mlx on Apple Silicon."
    warn "Proceeding anyway (experimental)..."
  fi

  if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — vLLM requires an NVIDIA GPU with CUDA."
  fi

  local CUDA_MAJOR="" TORCH_INDEX
  [[ -n "${NVCC_VERSION:-}" ]] && CUDA_MAJOR="$(echo "$NVCC_VERSION" | cut -d. -f1)"

  if [[ "$CUDA_MAJOR" == "11" ]]; then
    info "CUDA 11.x — cu118 torch"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  else
    info "CUDA 12.x / unknown — cu121 torch"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  fi

  pixi run pip install --quiet "vllm>=0.4" torch torchvision torchaudio \
    --index-url "$TORCH_INDEX"
  pixi run python -c "import vllm; print(f'vLLM {vllm.__version__} OK')" \
    || warn "vLLM import failed — check CUDA version compatibility"
  ok "vLLM install complete."
}
