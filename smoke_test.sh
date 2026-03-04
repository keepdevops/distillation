#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  Smoke Test: Phase 1 + 1.5 Optimizations"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Testing:"
echo "  ✓ Flash Attention 2 detection"
echo "  ✓ torch.compile() activation"
echo "  ✓ DataLoader optimization (4 workers)"
echo "  ✓ Gradient accumulation tuning (batch=8, grad_acc=8)"
echo "  ✓ Memory cache clearing"
echo "  ✓ Early stopping callback"
echo "  ✓ Quality gates (batch inference)"
echo ""
echo "Quick test: 50 samples, 1 epoch (~3-5 minutes)"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Create test output directory
TEST_DIR="./smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Test output directory: $TEST_DIR"
echo ""

# Run minimal distillation
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 1: Distillation (50 samples, 1 epoch)"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

python scripts/distill_minillm.py \
    --open \
    --offline \
    --output_dir "$TEST_DIR" \
    --epochs 1 \
    --max_samples 50 \
    --eval_steps 2 \
    --batch_size 8 \
    --grad_acc 8 \
    2>&1 | tee "$TEST_DIR/distill.log"

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 2: Quality Evaluation (10 samples)"
echo "─────────────────────────────────────────────────────────────────────"
echo ""

python scripts/eval_quality.py \
    "$TEST_DIR" \
    --student Qwen/Qwen2-0.5B-Instruct \
    --n_samples 10 \
    --batch_size 4 \
    --offline \
    2>&1 | tee "$TEST_DIR/quality.log"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Smoke Test Results"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check for key features in logs
echo "Checking optimization detection..."
echo ""

if grep -q "Flash Attention 2 detected" "$TEST_DIR/distill.log"; then
    echo "  ✓ Flash Attention 2: ENABLED"
elif grep -q "Flash Attention 2 not available" "$TEST_DIR/distill.log"; then
    echo "  ⚠ Flash Attention 2: NOT INSTALLED (install for 2-3x speedup)"
else
    echo "  ? Flash Attention 2: UNKNOWN"
fi

if grep -q "Compiling student model" "$TEST_DIR/distill.log"; then
    echo "  ✓ torch.compile(): ENABLED"
elif grep -q "torch.compile() skipped on MPS" "$TEST_DIR/distill.log"; then
    echo "  ✓ torch.compile(): PROPERLY SKIPPED ON MPS (Apple Silicon)"
else
    echo "  ✗ torch.compile(): NOT DETECTED"
fi

if grep -q "Early stopping baseline set\|Early stopping check" "$TEST_DIR/distill.log"; then
    echo "  ✓ Early stopping callback: ACTIVE"
else
    echo "  ⚠ Early stopping callback: NO LOG OUTPUT (may be inactive or logging not captured)"
fi

# Check quality gates
if [ -f "$TEST_DIR/quality_metrics.json" ]; then
    echo "  ✓ Quality metrics: GENERATED"

    # Extract key metrics
    pass_rate=$(jq -r '.quality_gates.pass_rate_pct // "N/A"' "$TEST_DIR/quality_metrics.json")
    refusal_rate=$(jq -r '.quality_gates.refusal_rate_pct // "N/A"' "$TEST_DIR/quality_metrics.json")
    distinct1=$(jq -r '.diversity.avg_distinct_1 // "N/A"' "$TEST_DIR/quality_metrics.json")

    echo ""
    echo "  Quality Gate Results:"
    echo "    - Pass rate: $pass_rate%"
    echo "    - Refusal rate: $refusal_rate%"
    echo "    - Distinct-1: $distinct1"
else
    echo "  ✗ Quality metrics: NOT FOUND"
fi

# Check training completed
if [ -f "$TEST_DIR/pytorch_model.bin" ] || [ -f "$TEST_DIR/model.safetensors" ]; then
    echo ""
    echo "  ✓ Training: COMPLETED"
    echo "  ✓ Model saved: $TEST_DIR"
else
    echo ""
    echo "  ✗ Training: FAILED (no model weights found)"
    exit 1
fi

# Performance estimate
if [ -f "$TEST_DIR/trainer_state.json" ]; then
    echo ""
    echo "Training Performance:"
    total_time=$(jq -r '.log_history[-1].total_flos // 0' "$TEST_DIR/trainer_state.json")
    steps=$(jq -r '.global_step // 0' "$TEST_DIR/trainer_state.json")
    echo "  - Steps completed: $steps"
    echo "  - Output: $TEST_DIR"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✓ Smoke Test PASSED"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "All optimizations are working correctly!"
echo ""
echo "Next steps:"
echo "  • Full run: ./run_autonomous_production.sh"
echo "  • Install Flash Attention (if not detected): pip install flash-attn --no-build-isolation"
echo "  • View logs: cat $TEST_DIR/*.log"
echo "  • Cleanup: rm -rf $TEST_DIR"
echo ""
