#!/usr/bin/env bash
# install.sh — Interactive setup for the distill project
#
# Usage:
#   ./scripts/install.sh                              # interactive wizard
#   ./scripts/install.sh --yes                        # non-interactive full install
#   ./scripts/install.sh --shared-models              # storage dirs + env vars only
#   ./scripts/install.sh --thermal-agent              # LaunchAgent only
#   ./scripts/install.sh --vllm                       # vLLM only (NVIDIA)
#   ./scripts/install.sh --full                       # full install, skip wizard
#
# Config flags (usable with any mode):
#   --model-path PATH    override MODEL_PATH  (default: /Users/Shared/llama/models)
#   --llama-root PATH    override LLAMA_CPP_ROOT (default: /Users/Shared/llama)
#   --threshold N        thermal pause threshold °C  (default: 85)
#   --interval N         thermal poll interval sec   (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# ── prompt helpers ────────────────────────────────────────────────────────────
prompt_yn() {   # prompt_yn "Question" y|n  →  returns 0 (yes) or 1 (no)
  local q="$1" def="${2:-y}" hint ans
  [[ "$def" == "y" ]] && hint="Y/n" || hint="y/N"
  read -r -p "  $q [$hint]: " ans </dev/tty
  [[ "${ans:-$def}" =~ ^[Yy] ]]
}

prompt_val() {  # prompt_val "Question" "default"  →  echoes chosen value
  local q="$1" def="$2" ans
  read -r -p "  $q [$def]: " ans </dev/tty
  echo "${ans:-$def}"
}

# ── arg parsing ───────────────────────────────────────────────────────────────
DO_FULL=false; DO_SHARED=false; DO_THERMAL=false; DO_VLLM=false
EXPLICIT=false; NONINTERACTIVE=false

MODEL_PATH="${MODEL_PATH:-/Users/Shared/llama/models}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/Users/Shared/llama}"
THRESHOLD=85; INTERVAL=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)           NONINTERACTIVE=true; DO_FULL=true; DO_SHARED=true ;;
    --full)             DO_FULL=true;   EXPLICIT=true ;;
    --shared-models)    DO_SHARED=true; EXPLICIT=true ;;
    --thermal-agent)    DO_THERMAL=true; EXPLICIT=true ;;
    --vllm)             DO_VLLM=true;   EXPLICIT=true ;;
    --model-path)       MODEL_PATH="$2"; shift ;;
    --model-path=*)     MODEL_PATH="${1#*=}" ;;
    --llama-root)       LLAMA_CPP_ROOT="$2"; shift ;;
    --llama-root=*)     LLAMA_CPP_ROOT="${1#*=}" ;;
    --threshold)        THRESHOLD="$2"; shift ;;
    --threshold=*)      THRESHOLD="${1#*=}" ;;
    --interval)         INTERVAL="$2"; shift ;;
    --interval=*)       INTERVAL="${1#*=}" ;;
    *) error "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

WIZARD=false
[[ "$EXPLICIT" == false && "$NONINTERACTIVE" == false ]] && WIZARD=true

# ── hardware detection ────────────────────────────────────────────────────────
detect_hardware() {
  OS="unknown"; ARCH="$(uname -m)"; RAM_GB=0
  IS_APPLE_SILICON=false; APPLE_CHIP=""; HAS_NVIDIA=false; NVCC_VERSION=""

  case "$(uname -s)" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    *)      error "Unsupported OS: $(uname -s)"; exit 1 ;;
  esac

  if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
    IS_APPLE_SILICON=true
    APPLE_CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    [[ -z "$APPLE_CHIP" ]] && \
      APPLE_CHIP="$(system_profiler SPHardwareDataType 2>/dev/null \
        | awk -F': ' '/Chip/{print $2}' | xargs || true)"
  fi

  if [[ "$OS" == "macos" ]]; then
    RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
  else
    RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
  fi

  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    HAS_NVIDIA=true
    NVCC_VERSION="$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || true)"
  fi

  PROFILE="cpu"
  $IS_APPLE_SILICON && PROFILE="apple_silicon"
  $HAS_NVIDIA       && PROFILE="nvidia"
}

# ── interactive wizard ────────────────────────────────────────────────────────
run_wizard() {
  echo -e "\n${BOLD}╔══════════════════════════════════╗${NC}"
  echo -e "${BOLD}║   Distill Installer Wizard  v1   ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════╝${NC}\n"
  echo "  Detected hardware:"
  echo "    OS      : $OS / $ARCH"
  [[ -n "$APPLE_CHIP" ]] && echo "    Chip    : $APPLE_CHIP"
  echo "    RAM     : ${RAM_GB} GB"
  echo "    Profile : $PROFILE"
  $HAS_NVIDIA && echo "    nvcc    : ${NVCC_VERSION:-unknown}"

  section "Configuration"
  MODEL_PATH="$(prompt_val "Model storage path (MODEL_PATH)" "$MODEL_PATH")"
  LLAMA_CPP_ROOT="$(prompt_val "llama.cpp root (LLAMA_CPP_ROOT)" "$LLAMA_CPP_ROOT")"

  section "What to install"
  echo "  1) Full install  — pixi env + packages + shared model storage  (recommended)"
  echo "  2) Shared model storage only"
  echo "  3) Thermal monitoring agent only"
  echo "  4) vLLM only  (NVIDIA)"
  echo "  5) Custom — choose components"
  echo ""
  local choice
  choice="$(prompt_val "Choice" "1")"

  case "$choice" in
    1) DO_FULL=true; DO_SHARED=true ;;
    2) DO_SHARED=true ;;
    3) DO_THERMAL=true ;;
    4) DO_VLLM=true ;;
    5)
      if prompt_yn "Full install (pixi env + packages)" "y"; then DO_FULL=true; fi
      if prompt_yn "Shared model storage setup"          "y"; then DO_SHARED=true; fi
      if prompt_yn "Thermal monitoring agent"            "n"; then DO_THERMAL=true; fi
      if prompt_yn "vLLM (NVIDIA only)"                 "n"; then DO_VLLM=true; fi
      ;;
    *) error "Invalid choice: $choice"; exit 1 ;;
  esac

  if [[ "$DO_THERMAL" == true ]]; then
    section "Thermal Agent"
    THRESHOLD="$(prompt_val "Pause threshold °C" "$THRESHOLD")"
    INTERVAL="$(prompt_val "Poll interval seconds" "$INTERVAL")"
  fi

  section "Summary"
  echo "  MODEL_PATH      : $MODEL_PATH"
  echo "  LLAMA_CPP_ROOT  : $LLAMA_CPP_ROOT"
  echo "  Full install    : $DO_FULL"
  echo "  Shared storage  : $DO_SHARED"
  echo "  Thermal agent   : $DO_THERMAL  (${THRESHOLD}°C / ${INTERVAL}s)"
  echo "  vLLM            : $DO_VLLM"
  echo ""
  if ! prompt_yn "Proceed?" "y"; then echo "Aborted."; exit 0; fi
}

# ── main ──────────────────────────────────────────────────────────────────────
detect_hardware

if [[ "$WIZARD" == true ]]; then run_wizard; fi

# Export for sourced helpers
export MODEL_PATH LLAMA_CPP_ROOT THRESHOLD INTERVAL REPO_ROOT SCRIPT_DIR
export OS ARCH PROFILE RAM_GB IS_APPLE_SILICON APPLE_CHIP HAS_NVIDIA NVCC_VERSION

# shellcheck source=install/components.sh
source "$SCRIPT_DIR/install/components.sh"
# shellcheck source=install/full.sh
source "$SCRIPT_DIR/install/full.sh"

if [[ "$DO_SHARED"  == true ]]; then install_shared_models; fi
if [[ "$DO_THERMAL" == true ]]; then install_thermal_agent; fi
if [[ "$DO_VLLM"    == true ]]; then install_vllm; fi
if [[ "$DO_FULL"    == true ]]; then run_full_install; fi
