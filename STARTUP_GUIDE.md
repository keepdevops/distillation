# Distillation Pipeline Startup Guide

Complete startup order for running the autonomous distillation pipeline.

---

## Terminal 1 — (Optional) Watchdog Monitor

Runs in background, detects loss plateau and thermal spikes. Start this first so it's watching when training begins.

```bash
cd /Users/caribou/distill
python scripts/training_watchdog.py \
  ./distilled-minillm \
  --config configs/watchdog_rules.json
```

**Note:** The output directory is a required positional argument (must come before `--config`).

---

## Terminal 2 — (Optional) Thermal Monitor

Log CPU/GPU temps while training. Start alongside the watchdog.

```bash
python scripts/monitor_cpu_gpu_temp.py \
  --interval 10 \
  --log ./thermal.log
```

**Optional:** Add `--fan-control` to enable automatic fan control (requires Macs Fan Control app).

---

## Terminal 3 — Autonomous Distillation Agent

This is the main script — runs the full pipeline end-to-end. Start this after the monitors are up.

### Minimal Run (Qwen2, pytorch backend)
```bash
python scripts/run_distillation_agent.py \
  --open \
  --epochs 2 \
  --max_samples 2000 \
  --export gguf \
  --log_experiment
```

### Full Autonomous Run (all features)
```bash
python scripts/run_distillation_agent.py \
  --open \
  --backend pytorch \
  --epochs 2 \
  --max_samples 2000 \
  --export gguf \
  --compare_teacher \
  --benchmarks \
  --log_experiment \
  --watchdog \
  --seed 42
```

### Multi-Trial Hyperparameter Search
```bash
python scripts/run_distillation_agent.py \
  --open \
  --n_trials 3 \
  --epochs 2 \
  --export gguf \
  --log_experiment
```

### Using Config File
```bash
# Edit configs/agent_config.json first, then:
python scripts/run_distillation_agent.py --config configs/agent_config.json
```

#### The Agent Pipeline Sequence

The agent internally runs these scripts in this order, automatically:

1. `experiment_log.py` — Reads history, prints past runs
2. `generate_synthetic_data.py` — (if `--synthetic_data`)
3. `distill_sft.py` — (if `--curriculum`)
4. `distill_minillm.py` — Main training
5. Export (gguf/mlx/coreml) — Model export
6. `run_eval.py` — Perplexity eval
7. `eval_quality.py` — Diversity + judge
8. `run_benchmarks.py` — (if `--benchmarks`) WikiText-2
9. `experiment_log.py` — Writes result to log

---

## Terminal 4 — Gradio Dashboard

Start this any time — before, during, or after training. It reads live metrics while the agent runs.

```bash
python scripts/dashboard.py \
  --runs_dir . \
  --port 7860
```

Then open http://127.0.0.1:7860 in your browser.

### Dashboard Tabs and Data Sources

| Tab         | Populated by                                      |
|-------------|---------------------------------------------------|
| Plots       | `metrics.jsonl` (live during training)            |
| Pipeline    | `trainer_state.json` + GGUF files (after export)  |
| Thermal     | `thermal.log` (if monitor running)                |
| Evaluate    | Auto-discovers models from HF cache + output dirs |
| Quality     | `quality_metrics.json` (after `eval_quality.py`)  |
| Experiments | `experiment_log.jsonl` (after `--log_experiment`) |

---

## Verification

All scripts verified as of 2026-03-03:

- ✓ All 11 referenced scripts exist
- ✓ Config files exist: `configs/watchdog_rules.json`, `configs/agent_config.json`
- ✓ All command-line arguments verified
- ✓ All Python files compile successfully

---

## Common Output Directories

Default output structure:
```
distill/
├── distilled-minillm/              # Training output
│   ├── trainer_state.json          # Live training state
│   ├── metrics.jsonl               # Eval metrics
│   ├── checkpoint-*/               # Model checkpoints
│   ├── *.gguf                      # Exported GGUF files
│   ├── quality_metrics.json        # Quality eval results
│   └── watchdog_suggestions.json   # Watchdog output
├── thermal.log                     # Thermal monitoring CSV
├── experiment_log.jsonl            # Multi-run history
└── configs/
    ├── agent_config.json
    └── watchdog_rules.json
```

---

## Troubleshooting

### Watchdog Not Starting
- Ensure output directory exists: `mkdir -p ./distilled-minillm`
- Check config file syntax: `python -m json.tool configs/watchdog_rules.json`

### Thermal Monitor Fails
- Install mactop: `brew install mactop`
- For fan control: `brew install --cask macs-fan-control`

### Dashboard Shows No Models
- Check `--runs_dir` points to parent directory containing training outputs
- Verify HF cache exists: `ls ~/.cache/huggingface/hub`

### Agent Fails to Start
- Check offline mode if air-gapped: add `--offline`
- Verify HF cache populated: `python scripts/cache_models.py --open`
