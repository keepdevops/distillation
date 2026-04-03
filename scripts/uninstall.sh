#!/usr/bin/env bash
# uninstall.sh — Remove components installed by install.sh
#
# Usage:
#   ./scripts/uninstall.sh                # interactive wizard
#   ./scripts/uninstall.sh --yes          # remove all (does NOT delete model data)
#   ./scripts/uninstall.sh --thermal-agent
#   ./scripts/uninstall.sh --zshrc
#   ./scripts/uninstall.sh --pixi
#   ./scripts/uninstall.sh --brew-tools
#   ./scripts/uninstall.sh --model-dirs   # also remove shared model directories (destructive)
#
# Config:
#   --model-path PATH    override MODEL_PATH  (default: /Users/Shared/llama/models/models)
#   --llama-root PATH    override LLAMA_CPP_ROOT (default: /Users/Shared/llama)

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
prompt_yn() {
  local q="$1" def="${2:-y}" hint ans
  [[ "$def" == "y" ]] && hint="Y/n" || hint="y/N"
  read -r -p "  $q [$hint]: " ans </dev/tty
  [[ "${ans:-$def}" =~ ^[Yy] ]]
}

# ── arg parsing ───────────────────────────────────────────────────────────────
DO_THERMAL=false; DO_ZSHRC=false; DO_PIXI=false
DO_BREW=false; DO_MODEL_DIRS=false
EXPLICIT=false; NONINTERACTIVE=false

MODEL_PATH="${MODEL_PATH:-/Users/Shared/llama/models/models}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/Users/Shared/llama}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)         NONINTERACTIVE=true
                      DO_THERMAL=true; DO_ZSHRC=true; DO_PIXI=true; DO_BREW=true ;;
    --thermal-agent)  DO_THERMAL=true;    EXPLICIT=true ;;
    --zshrc)          DO_ZSHRC=true;      EXPLICIT=true ;;
    --pixi)           DO_PIXI=true;       EXPLICIT=true ;;
    --brew-tools)     DO_BREW=true;       EXPLICIT=true ;;
    --model-dirs)     DO_MODEL_DIRS=true; EXPLICIT=true ;;
    --model-path)     MODEL_PATH="$2";    shift ;;
    --model-path=*)   MODEL_PATH="${1#*=}" ;;
    --llama-root)     LLAMA_CPP_ROOT="$2"; shift ;;
    --llama-root=*)   LLAMA_CPP_ROOT="${1#*=}" ;;
    *) error "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

WIZARD=false
[[ "$EXPLICIT" == false && "$NONINTERACTIVE" == false ]] && WIZARD=true

# ── interactive wizard ────────────────────────────────────────────────────────
run_wizard() {
  echo -e "\n${BOLD}╔════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║   Distill Uninstaller  v1          ║${NC}"
  echo -e "${BOLD}╚════════════════════════════════════╝${NC}\n"
  echo "  This removes components installed by install.sh."
  echo "  Model data is never deleted unless you choose --model-dirs."
  echo ""

  section "Select components to remove"
  if prompt_yn "Thermal monitoring LaunchAgent" "y"; then DO_THERMAL=true; fi
  if prompt_yn "~/.zshrc env vars (MODEL_PATH, LLAMA_CPP_ROOT)" "y"; then DO_ZSHRC=true; fi
  if prompt_yn "pixi environment (.pixi/ cache)" "y"; then DO_PIXI=true; fi
  if prompt_yn "Homebrew tools (mactop, llama.cpp)" "n"; then DO_BREW=true; fi

  warn "Model directories contain your downloaded models."
  warn "Only choose yes if you want to delete those files permanently."
  if prompt_yn "Remove shared model directories (DESTRUCTIVE)" "n"; then
    DO_MODEL_DIRS=true
  fi

  section "Summary"
  echo "  Thermal agent  : $DO_THERMAL"
  echo "  ~/.zshrc vars  : $DO_ZSHRC"
  echo "  pixi env       : $DO_PIXI"
  echo "  Brew tools     : $DO_BREW"
  echo "  Model dirs     : $DO_MODEL_DIRS"
  echo ""
  if ! prompt_yn "Proceed?" "y"; then echo "Aborted."; exit 0; fi
}

# ── component removers ────────────────────────────────────────────────────────
remove_thermal_agent() {
  section "Thermal Monitoring Agent"
  local PLIST="$HOME/Library/LaunchAgents/com.distillation.thermal_agent.plist"

  if launchctl list 2>/dev/null | grep -q com.distillation.thermal_agent; then
    launchctl unload "$PLIST" 2>/dev/null && ok "Agent unloaded" \
      || warn "launchctl unload failed (agent may already be stopped)"
  else
    info "Agent not currently loaded — skipping unload"
  fi

  if [[ -f "$PLIST" ]]; then
    rm "$PLIST"
    ok "Removed $PLIST"
  else
    info "Plist not found — already removed"
  fi

  for log in "$REPO_ROOT/thermal_agent.jsonl" \
             "$REPO_ROOT/thermal_agent.stdout.log" \
             "$REPO_ROOT/thermal_agent.stderr.log"; do
    [[ -f "$log" ]] && rm "$log" && info "Removed $(basename "$log")" || true
  done
}

remove_zshrc_vars() {
  section "~/.zshrc Environment Variables"
  local ZSHRC="$HOME/.zshrc"
  [[ -f "$ZSHRC" ]] || { info "~/.zshrc not found — nothing to do"; return 0; }

  local changed=false

  # Remove the Distilled model storage comment block + MODEL_PATH line
  if grep -q "Distilled model storage" "$ZSHRC"; then
    sed -i '' '/# Distilled model storage/d' "$ZSHRC"
    ok "Removed '# Distilled model storage' comment"
    changed=true
  fi

  if grep -q "export MODEL_PATH=" "$ZSHRC"; then
    sed -i '' '/export MODEL_PATH=/d' "$ZSHRC"
    ok "Removed MODEL_PATH from ~/.zshrc"
    changed=true
  fi

  if grep -q "export LLAMA_CPP_ROOT=" "$ZSHRC"; then
    sed -i '' '/export LLAMA_CPP_ROOT=/d' "$ZSHRC"
    ok "Removed LLAMA_CPP_ROOT from ~/.zshrc"
    changed=true
  fi

  $changed || info "No distill env vars found in ~/.zshrc"
  $changed && info "Reload: source ~/.zshrc"
}

remove_pixi_env() {
  section "pixi Environment"
  local PIXI_DIR="$REPO_ROOT/.pixi"

  if [[ -d "$PIXI_DIR" ]]; then
    if command -v pixi &>/dev/null; then
      (cd "$REPO_ROOT" && pixi clean 2>/dev/null) && ok "pixi environment cleaned" \
        || { warn "pixi clean failed — removing .pixi/ manually"; rm -rf "$PIXI_DIR"; ok "Removed $PIXI_DIR"; }
    else
      rm -rf "$PIXI_DIR"
      ok "Removed $PIXI_DIR"
    fi
  else
    info ".pixi/ directory not found — already removed"
  fi
}

remove_brew_tools() {
  section "Homebrew Tools"
  [[ "$(uname -s)" != "Darwin" ]] && { info "Not macOS — skipping"; return 0; }

  if ! command -v brew &>/dev/null; then
    warn "Homebrew not found — nothing to remove"
    return 0
  fi

  if brew list mactop &>/dev/null 2>&1; then
    brew uninstall mactop && ok "mactop removed" || warn "mactop uninstall failed"
  else
    info "mactop not installed via brew — skipping"
  fi

  if brew list llama.cpp &>/dev/null 2>&1; then
    warn "llama.cpp is also used by other tools."
    if [[ "$NONINTERACTIVE" == false ]] && ! prompt_yn "Uninstall llama.cpp via brew?" "n"; then
      info "Skipping llama.cpp"
    else
      brew uninstall llama.cpp && ok "llama.cpp removed" || warn "llama.cpp uninstall failed"
    fi
  else
    info "llama.cpp not installed via brew — skipping"
  fi
}

remove_model_dirs() {
  section "Shared Model Directories"
  warn "About to delete model files. This cannot be undone."

  if [[ "$NONINTERACTIVE" == false ]]; then
    if ! prompt_yn "Really delete $MODEL_PATH and $LLAMA_CPP_ROOT/models?" "n"; then
      info "Skipping model directory removal"
      return 0
    fi
  fi

  if [[ -d "$MODEL_PATH" ]]; then
    rm -rf "$MODEL_PATH"
    ok "Removed $MODEL_PATH"
  else
    info "$MODEL_PATH not found"
  fi

  local LLAMA_MODELS="$LLAMA_CPP_ROOT/models"
  if [[ -d "$LLAMA_MODELS" ]]; then
    rm -rf "$LLAMA_MODELS"
    ok "Removed $LLAMA_MODELS"
  else
    info "$LLAMA_MODELS not found"
  fi
}

# ── main ──────────────────────────────────────────────────────────────────────
[[ "$WIZARD" == true ]] && run_wizard

[[ "$DO_THERMAL" == true ]]    && remove_thermal_agent
[[ "$DO_ZSHRC" == true ]]      && remove_zshrc_vars
[[ "$DO_PIXI" == true ]]       && remove_pixi_env
[[ "$DO_BREW" == true ]]       && remove_brew_tools
[[ "$DO_MODEL_DIRS" == true ]] && remove_model_dirs

section "Done"
ok "Uninstall complete."
