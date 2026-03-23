#!/usr/bin/env bash
# install.sh — Detect hardware and install distill dependencies accordingly
# Supports: macOS (Apple Silicon / Intel), Linux (NVIDIA / CPU)
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Detect OS ──────────────────────────────────────────────────────────────
section "Detecting System"

OS="unknown"
case "$(uname -s)" in
  Darwin) OS="macos" ;;
  Linux)  OS="linux" ;;
  *)      error "Unsupported OS: $(uname -s)"; exit 1 ;;
esac
info "OS: $OS"

# ── 2. Detect Architecture ───────────────────────────────────────────────────
ARCH="$(uname -m)"
info "Architecture: $ARCH"

# ── 3. Detect Apple Silicon chip generation ───────────────────────────────────
APPLE_CHIP=""
IS_APPLE_SILICON=false
if [[ "$OS" == "macos" ]]; then
  if [[ "$ARCH" == "arm64" ]]; then
    IS_APPLE_SILICON=true
    # sysctl returns something like "Apple M3 Max"
    CHIP_RAW="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    if [[ -z "$CHIP_RAW" ]]; then
      CHIP_RAW="$(system_profiler SPHardwareDataType 2>/dev/null | grep 'Chip' | awk -F': ' '{print $2}' | xargs || true)"
    fi
    APPLE_CHIP="$CHIP_RAW"
    ok "Apple Silicon detected: $APPLE_CHIP"
  else
    warn "macOS with Intel CPU — MLX not available"
  fi
fi

# ── 4. Detect RAM ─────────────────────────────────────────────────────────────
RAM_GB=0
if [[ "$OS" == "macos" ]]; then
  RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
elif [[ "$OS" == "linux" ]]; then
  RAM_KB="$(grep MemTotal /proc/meminfo | awk '{print $2}')"
  RAM_GB=$(( RAM_KB / 1024 / 1024 ))
fi
info "RAM: ${RAM_GB} GB"

# ── 5. Detect NVIDIA GPU (Linux / Windows) ───────────────────────────────────
HAS_NVIDIA=false
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
  if nvidia-smi &>/dev/null 2>&1; then
    HAS_NVIDIA=true
    CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    NVCC_VERSION="$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || true)"
    ok "NVIDIA GPU detected (driver: $CUDA_VERSION, nvcc: ${NVCC_VERSION:-not found})"
  fi
fi

# ── 6. Summarise profile ──────────────────────────────────────────────────────
section "Installation Profile"

PROFILE="cpu"
if $IS_APPLE_SILICON; then
  PROFILE="apple_silicon"
elif $HAS_NVIDIA; then
  PROFILE="nvidia"
fi

echo "  Profile  : $PROFILE"
echo "  RAM      : ${RAM_GB} GB"
if [[ -n "$APPLE_CHIP" ]]; then
  echo "  Chip     : $APPLE_CHIP"
fi
if $HAS_NVIDIA; then
  echo "  CUDA     : $CUDA_VERSION"
fi

# ── 7. Check / Install pixi ──────────────────────────────────────────────────
section "Package Manager"

if command -v pixi &>/dev/null; then
  ok "pixi found: $(pixi --version)"
else
  info "Installing pixi..."
  curl -fsSL https://pixi.sh/install.sh | bash
  # shellcheck source=/dev/null
  export PATH="$HOME/.pixi/bin:$PATH"
  ok "pixi installed"
fi

# ── 8. Homebrew tools (macOS only) ───────────────────────────────────────────
if [[ "$OS" == "macos" ]]; then
  section "macOS Tools"

  if ! command -v brew &>/dev/null; then
    warn "Homebrew not found — skipping brew packages (install from https://brew.sh)"
  else
    ok "Homebrew: $(brew --version | head -1)"

    # mactop — thermal monitoring (no sudo, works on M-series)
    if ! command -v mactop &>/dev/null; then
      info "Installing mactop..."
      brew install mactop && ok "mactop installed" || warn "mactop install failed (non-fatal)"
    else
      ok "mactop: already installed"
    fi

    # llama.cpp — GGUF conversion / inference server
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
  fi
fi

# ── 9. Core pixi environment ──────────────────────────────────────────────────
section "Core Environment (pixi)"
cd "$SCRIPT_DIR"
info "Running: pixi install"
pixi install
ok "pixi environment ready"

# ── 10. Python packages via pip inside pixi ───────────────────────────────────
section "Python Packages"

# Helper to run pip inside the pixi environment
pix_pip() { pixi run pip install --quiet "$@"; }

# ── Common packages ────────────────────────────────────────────────────────────
info "Installing common packages..."
pix_pip \
  "transformers>=4.38" \
  "datasets>=2.14" \
  "accelerate>=0.24" \
  peft \
  evaluate \
  "trl>=0.10" \
  "gradio>=4.0" \
  matplotlib seaborn tqdm rich psutil

# ── Profile-specific packages ─────────────────────────────────────────────────
case "$PROFILE" in

  apple_silicon)
    section "Apple Silicon Packages"

    # PyTorch — ships with MPS support on macOS arm64
    info "Installing PyTorch (MPS backend)..."
    pix_pip torch torchvision torchaudio
    ok "PyTorch installed (MPS enabled)"

    # MLX — Apple Silicon native framework
    info "Installing MLX..."
    pix_pip mlx "mlx-lm>=0.20"
    ok "MLX installed"

    # CoreML export
    info "Installing coremltools..."
    pix_pip "coremltools>=8.0"
    ok "coremltools installed"

    # Recommend larger / smaller models based on RAM
    echo ""
    if (( RAM_GB >= 32 )); then
      ok "RAM: ${RAM_GB} GB — suitable for 7B+ teachers with MLX"
    elif (( RAM_GB >= 16 )); then
      warn "RAM: ${RAM_GB} GB — stick to ≤3B teachers; use Q4 quants"
    else
      warn "RAM: ${RAM_GB} GB — use 1B student + 1.5B teacher only"
    fi
    ;;

  nvidia)
    section "NVIDIA / CUDA Packages"

    # Pick torch index URL based on detected CUDA driver version
    # nvidia-smi driver version e.g. "525.105.17"; map to CUDA toolkit version
    CUDA_MAJOR=""
    if [[ -n "$NVCC_VERSION" ]]; then
      CUDA_MAJOR="$(echo "$NVCC_VERSION" | cut -d. -f1)"
    fi

    if [[ "$CUDA_MAJOR" == "12" ]]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      info "CUDA 12.x detected — installing PyTorch cu121"
    elif [[ "$CUDA_MAJOR" == "11" ]]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu118"
      info "CUDA 11.x detected — installing PyTorch cu118"
    else
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      warn "CUDA version undetermined — defaulting to cu121"
    fi

    pix_pip torch torchvision torchaudio --index-url "$TORCH_INDEX"
    ok "PyTorch (CUDA) installed"

    # bitsandbytes — 4-bit / 8-bit quantisation
    info "Installing bitsandbytes..."
    pix_pip bitsandbytes
    ok "bitsandbytes installed"

    if (( RAM_GB >= 40 )); then
      ok "VRAM/RAM profile: suitable for large teacher models"
    fi
    ;;

  cpu)
    section "CPU-only Packages"
    warn "No accelerator detected — installing CPU-only PyTorch (slow for training)"
    pix_pip torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ok "PyTorch (CPU) installed"
    ;;
esac

# ── 11. Verify installation ───────────────────────────────────────────────────
section "Verification"

pixi run python - <<'PYEOF'
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")

if torch.backends.mps.is_available():
    print("MPS:     available")
elif torch.cuda.is_available():
    print(f"CUDA:    available ({torch.cuda.get_device_name(0)})")
else:
    print("Accel:   CPU only")

try:
    import mlx.core as mx
    print(f"MLX:     {mx.__version__}")
except ImportError:
    pass

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except ImportError:
    print("transformers: NOT installed")

try:
    import gradio
    print(f"gradio:  {gradio.__version__}")
except ImportError:
    print("gradio:  NOT installed")
PYEOF

ok "Verification complete"

# ── 12. Done ──────────────────────────────────────────────────────────────────
section "Setup Complete"
echo ""
echo "  Profile  : $PROFILE"
if [[ -n "$APPLE_CHIP" ]]; then
  echo "  Chip     : $APPLE_CHIP"
fi
echo "  RAM      : ${RAM_GB} GB"
echo ""
echo "  Quick-start:"
echo "    pixi run python scripts/run_distillation_agent.py --open --backend mlx"
echo "    pixi run python scripts/eval_gradio.py"
echo ""
ok "Done!"
