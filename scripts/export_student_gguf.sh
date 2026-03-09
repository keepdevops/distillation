#!/usr/bin/env bash
#
# Convert distilled student (HuggingFace) to GGUF for llama.cpp.
# Requires llama.cpp cloned; run from project root.
#
# Usage:
#   ./scripts/export_student_gguf.sh [student_dir] [llama_cpp_dir]
#
# Example:
#   ./scripts/export_student_gguf.sh ./distilled-minillm ../llama.cpp

set -e

STUDENT_DIR="${1:-./distilled-minillm}"
# llama.cpp: ./llama.cpp (if cloned into project) or ../llama.cpp
LLAMA_CPP="${2}"
if [[ -z "$LLAMA_CPP" ]]; then
  if [[ -d "/Users/Shared/llama" ]]; then
    LLAMA_CPP="/Users/Shared/llama"
  elif [[ -d "./llama.cpp" ]]; then
    LLAMA_CPP="./llama.cpp"
  else
    LLAMA_CPP="../llama.cpp"
  fi
fi
OUTTYPE="${OUTTYPE:-f16}"

if [[ ! -d "$STUDENT_DIR" ]]; then
  echo "Error: Student dir not found: $STUDENT_DIR"
  echo "Run distill_minillm.py first, or pass path: ./export_student_gguf.sh /path/to/distilled-minillm"
  exit 1
fi

CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT" ]]; then
  echo "Error: convert_hf_to_gguf.py not found at $CONVERT"
  echo "Clone: git clone https://github.com/ggerganov/llama.cpp"
  echo "  into project: distill/llama.cpp"
  echo "  or sibling: ../llama.cpp"
  exit 1
fi

STUDENT_ABS=$(cd "$STUDENT_DIR" && pwd)
OUT_NAME=$(basename "$STUDENT_ABS")-${OUTTYPE}.gguf
GGUF_DIR="/Users/Shared/llama/models"
mkdir -p "$GGUF_DIR"
OUT_FILE="$GGUF_DIR/$OUT_NAME"

echo "Converting $STUDENT_ABS -> $OUT_FILE (outtype=$OUTTYPE)"
python "$CONVERT" "$STUDENT_ABS" --outfile "$OUT_FILE" --outtype "$OUTTYPE"
echo "Done. Run: /Users/Shared/llama/llama-server -m $OUT_FILE"
