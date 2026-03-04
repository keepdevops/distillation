#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  Autonomous Production Distillation"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Trials: 5 (autonomous hyperparameter search)"
echo "  • Curriculum: Teacher-generated SFT warmup (1 epoch)"
echo "  • Synthetic data: 1500 samples augmentation"
echo "  • Benchmarks: WikiText-2 forgetting detection"
echo "  • Quality gates: Batch inference, refusal detection, diversity"
echo "  • Judge: LLM-as-judge scoring enabled"
echo "  • Export: GGUF format (best trial only)"
echo ""
echo "Optimizations (Phase 1 + 1.5 - 50-60% faster):"
echo "  • Flash Attention 2: 2-3x attention speedup (auto-detected)"
echo "  • torch.compile(): 20-40% overall speedup (auto-enabled)"
echo "  • DataLoader: 4 workers, prefetch=2 (5-10% faster)"
echo "  • Gradient accumulation: batch=8, grad_acc=8 (10-15% faster)"
echo "  • Memory management: Cache clearing between stages"
echo "  • Early stopping: Diverging trials stop at step 20 (saves 55 min)"
echo "  • Quality eval: Winner only (saves 20-40 min)"
echo ""
echo "Expected timeline (with all optimizations):"
echo "  • Trial 1: ~50 min (includes teacher generation + compilation)"
echo "  • Trials 2-5: ~40 min each (one may stop early)"
echo "  • Total: 3.0-3.5 hours (was 7.2 hours, saved 3.7-4.2 hours!)"
echo ""
echo "Without Flash Attention (if not installed):"
echo "  • Total: 4.5-5.0 hours (still 2-2.5 hours saved)"
echo ""
echo "Monitoring:"
echo "  • Watchdog: Plateau detection (training progress)"
echo "  • Thermal agent: Optional system-wide protection (see THERMAL_AGENT.md)"
echo "  • Live logs: tail -f distilled-minillm/logs/*.log"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

python scripts/run_distillation_agent.py \
    --open \
    --offline \
    --n_trials 5 \
    --curriculum \
    --sft_epochs 1 \
    --epochs 2 \
    --synthetic_data \
    --n_synthetic 1500 \
    --benchmarks \
    --compare_teacher \
    --export gguf \
    --log_experiment \
    --watchdog

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✓ Autonomous search complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  • Experiment log: python scripts/experiment_log.py --show 5"
echo "  • Quality metrics: cat distilled-minillm/quality_metrics.json | jq"
echo "  • Winner model: ls -lh distilled-minillm/*.gguf"
echo ""
echo "Next steps:"
echo "  • Test model: python -m transformers.pipelines --model ./distilled-minillm"
echo "  • Dashboard: python scripts/dashboard.py --runs_dir . --port 7860"
echo "  • Serve: cd llama.cpp && ./build/bin/llama-server -m ../distill/distilled-minillm/*.gguf"
echo ""
