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
echo "Expected timeline:"
echo "  • Trial 1: ~2 hours (includes teacher generation)"
echo "  • Trials 2-5: ~1.3 hours each (teacher cached)"
echo "  • Total: 6.5-7 hours"
echo ""
echo "Monitoring:"
echo "  • Watchdog: Plateau detection + thermal monitoring"
echo "  • Fan control: Activates at 60°C"
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
    --judge \
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
