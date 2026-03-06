#!/bin/bash
# Setup script for shared model storage
# Run this on each user profile that needs access to models

set -e

MODEL_PATH="/Users/Shared/models"

echo "================================================"
echo "  Shared Model Storage Setup"
echo "================================================"
echo ""

# Check if directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Creating $MODEL_PATH..."
    mkdir -p "$MODEL_PATH"
    chmod 755 "$MODEL_PATH"
fi

echo "✓ Shared model directory: $MODEL_PATH"
ls -lh "$MODEL_PATH" 2>/dev/null | tail -n +2 | awk '{print "  " $0}'

# Add to .zshrc if not already there
ZSHRC="$HOME/.zshrc"
if [ -f "$ZSHRC" ]; then
    if ! grep -q "MODEL_PATH=/Users/Shared/models" "$ZSHRC"; then
        echo "" >> "$ZSHRC"
        echo "# Model storage for distillation" >> "$ZSHRC"
        echo "export MODEL_PATH=/Users/Shared/models" >> "$ZSHRC"
        echo "✓ Added MODEL_PATH to ~/.zshrc"
    else
        echo "✓ MODEL_PATH already in ~/.zshrc"
    fi
else
    echo "⚠  No .zshrc found, creating one..."
    echo "export MODEL_PATH=/Users/Shared/models" > "$ZSHRC"
    echo "✓ Created ~/.zshrc with MODEL_PATH"
fi

# Add to .bash_profile if it exists
BASH_PROFILE="$HOME/.bash_profile"
if [ -f "$BASH_PROFILE" ]; then
    if ! grep -q "MODEL_PATH=/Users/Shared/models" "$BASH_PROFILE"; then
        echo "" >> "$BASH_PROFILE"
        echo "# Model storage for distillation" >> "$BASH_PROFILE"
        echo "export MODEL_PATH=/Users/Shared/models" >> "$BASH_PROFILE"
        echo "✓ Added MODEL_PATH to ~/.bash_profile"
    else
        echo "✓ MODEL_PATH already in ~/.bash_profile"
    fi
fi

# Export for current session
export MODEL_PATH="/Users/Shared/models"

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Usage:"
echo "  1. Reload shell: source ~/.zshrc"
echo "  2. Or start a new terminal"
echo ""
echo "Test it:"
echo "  echo \$MODEL_PATH"
echo "  ls \$MODEL_PATH"
echo ""
echo "MLX example:"
echo "  mlx_lm.generate --model-dir \$MODEL_PATH/hf_cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/* --prompt 'Hello'"
echo ""
echo "Gradio UI:"
echo "  python scripts/eval_gradio.py --model_path \$MODEL_PATH/distilled-minillm"
echo ""
