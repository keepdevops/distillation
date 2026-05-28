#!/usr/bin/env bash
# install/full.sh — Full environment installer (sourced by install.sh)
# Requires: REPO_ROOT, OS, PROFILE, RAM_GB, IS_APPLE_SILICON, HAS_NVIDIA, NVCC_VERSION
# Requires: MODEL_PATH, LLAMA_CPP_ROOT
# Requires: color helpers (ok, warn, error, info, section) from install.sh

run_full_install() {
  _install_pixi
  _install_brew_tools
  _install_core_env
  _install_python_packages
  _install_profile_packages
  _verify_install
  _print_summary
}

_install_pixi() {
  section "Package Manager"
  if command -v pixi &>/dev/null; then
    ok "pixi: $(pixi --version)"
  else
    info "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
    ok "pixi installed"
  fi
}

_install_brew_tools() {
  [[ "${OS:-}" != "macos" ]] && return 0
  section "macOS Tools"

  if ! command -v brew &>/dev/null; then
    warn "Homebrew not found — skipping brew packages (install from https://brew.sh)"
    return 0
  fi
  ok "Homebrew: $(brew --version | head -1)"

  if ! command -v mactop &>/dev/null; then
    info "Installing mactop..."
    brew install mactop && ok "mactop installed" || warn "mactop install failed (non-fatal)"
  else
    ok "mactop: already installed"
  fi

  if [[ ! -d "/Users/Shared/llama" ]]; then
    if brew list llama.cpp &>/dev/null 2>&1; then
      ok "llama.cpp: already installed via brew"
    else
      info "Installing llama.cpp via brew..."
      brew install llama.cpp && ok "llama.cpp installed" || warn "llama.cpp install failed (non-fatal)"
    fi
  else
    ok "llama.cpp: found at /Users/Shared/llama"
  fi
}

_install_core_env() {
  section "Core Environment (pixi)"
  cd "$REPO_ROOT"
  info "Running: pixi install"
  pixi install
  ok "pixi environment ready"
}

_pix_pip() { pixi run pip install --quiet "$@"; }

_install_python_packages() {
  section "Python Packages"

  info "Common packages..."
  _pix_pip \
    "transformers>=4.38" "datasets>=2.14" "accelerate>=0.24" \
    peft evaluate "trl>=0.10" "gradio>=6.0" \
    matplotlib seaborn tqdm rich psutil

  info "llama-cpp-python (native build)..."
  (
    unset CMAKE_TOOLCHAIN_FILE CC CXX || true
    if ${IS_APPLE_SILICON:-false}; then
      CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 \
        pixi run pip install --quiet "llama-cpp-python>=0.2"
    else
      pixi run pip install --quiet "llama-cpp-python>=0.2"
    fi
  ) || warn "llama-cpp-python build failed — GGUF backend unavailable (non-fatal)"
}

_install_profile_packages() {
  case "${PROFILE:-cpu}" in

    apple_silicon)
      section "Apple Silicon Packages"
      info "PyTorch (MPS)..."
      _pix_pip torch torchvision torchaudio
      ok "PyTorch (MPS) installed"
      info "MLX..."
      _pix_pip mlx "mlx-lm>=0.20"
      ok "MLX installed"
      info "coremltools..."
      _pix_pip "coremltools>=8.0"
      ok "coremltools installed"
      if (( ${RAM_GB:-0} >= 32 )); then
        ok "RAM ${RAM_GB} GB — suitable for 7B+ teachers"
      else
        warn "RAM ${RAM_GB} GB — use ≤3B teachers with Q4 quants"
      fi
      ;;

    nvidia)
      section "NVIDIA / CUDA Packages"
      local CUDA_MAJOR="" TORCH_INDEX
      [[ -n "${NVCC_VERSION:-}" ]] && CUDA_MAJOR="$(echo "$NVCC_VERSION" | cut -d. -f1)"
      if [[ "$CUDA_MAJOR" == "12" ]]; then
        info "CUDA 12.x — cu121"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      elif [[ "$CUDA_MAJOR" == "11" ]]; then
        info "CUDA 11.x — cu118"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
      else
        warn "CUDA version unknown — defaulting to cu121"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      fi
      _pix_pip torch torchvision torchaudio --index-url "$TORCH_INDEX"
      ok "PyTorch (CUDA) installed"
      info "bitsandbytes + vLLM..."
      _pix_pip bitsandbytes "vllm>=0.4"
      ok "bitsandbytes + vLLM installed"
      ;;

    cpu)
      section "CPU-only Packages"
      warn "No accelerator — CPU PyTorch (slow for training)"
      _pix_pip torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
      ok "PyTorch (CPU) installed"
      ;;
  esac
}

_verify_install() {
  section "Verification"
  pixi run python - <<'PYEOF'
import sys, importlib
print(f"  Python: {sys.version.split()[0]}")
checks = [
  ("torch",        lambda m: f"PyTorch {m.__version__} | MPS={'yes' if m.backends.mps.is_available() else 'no'}"),
  ("mlx.core",     lambda m: f"MLX {m.__version__}"),
  ("transformers", lambda m: f"transformers {m.__version__}"),
  ("gradio",       lambda m: f"gradio {m.__version__}"),
  ("llama_cpp",    lambda m: f"llama-cpp-python {m.__version__}"),
  ("vllm",         lambda m: f"vLLM {m.__version__}"),
]
for pkg, fmt in checks:
  try:
    print(f"  {fmt(importlib.import_module(pkg))}")
  except ImportError:
    pass
PYEOF
  ok "Verification complete"
}

_print_summary() {
  section "Setup Complete"
  echo ""
  echo "  Profile       : ${PROFILE:-?}"
  [[ -n "${APPLE_CHIP:-}" ]] && echo "  Chip          : $APPLE_CHIP"
  echo "  RAM           : ${RAM_GB:-?} GB"
  echo "  MODEL_PATH    : $MODEL_PATH"
  echo "  LLAMA_CPP_ROOT: $LLAMA_CPP_ROOT"
  echo ""
  echo "  Quick-start:"
  echo "    ./scripts/gradio.sh                            # eval UI (auto-detects model)"
  echo "    ./scripts/gradio.sh --model_path <path>.gguf  # specific GGUF"
  echo "    ./scripts/run.sh production                   # full distillation run"
  echo ""
  ok "Done!"
}
