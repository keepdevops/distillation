# Distillation Pipeline Startup Guide

Complete startup order for running the autonomous distillation pipeline.

---

## Quick Option — All Services via `start.sh` (tmux)

`start.sh` launches the watchdog, dashboard, and optional thermal monitor in a single tmux session (`distill`) with `caffeinate` so the Mac stays awake:

```bash
./scripts/start.sh              # watchdog + dashboard
./scripts/start.sh --monitor    # + thermal monitor window
./scripts/stop.sh               # stop everything (kills tmux sessions)
```

Attach to inspect live output: `tmux attach -t distill`

Requires `tmux`: `brew install tmux`

Then skip to **Terminal 3** below to launch the distillation agent.

---

## Manual Startup (Terminal-by-Terminal)

---

## Terminal 1 — (Optional) Watchdog Monitor

Runs in background, detects loss plateau and divergence. Start this first so it's watching when training begins.

```bash
cd /Users/caribou/distill
python -m distill.training_watchdog \
  ./distilled-mlx \
  --config configs/watchdog_rules.json
```

**Note:** The output directory is a required positional argument (must come before `--config`). Use whatever `--output_dir` you pass to the agent (e.g. `./distilled-mlx` for MLX, `./distilled-minillm` for PyTorch).

---

## Terminal 2 — (Optional) Thermal Protection

Two options — choose one:

### Option A: Thermal Agent (recommended — system-wide protection + auto pause)

The thermal agent monitors all GPU workloads and writes `pause.flag` when temps exceed the threshold (85°C default), then clears it when temps recover.

```bash
python -m distill.thermal_agent \
  --watch . \
  --threshold 85 \
  --interval 30
```

Install as always-on LaunchAgent (survives reboots):

```bash
./scripts/install_thermal_agent.sh
```

### Option B: Thermal Logger (logging only — no auto-pause)

Records CPU/GPU temps to a log file for the dashboard's Thermal tab. No protection.

```bash
python -m distill.monitor_cpu_gpu_temp \
  --interval 10 \
  --log ./thermal.log
```

Add `--fan-control` to enable fan control integration (requires Macs Fan Control app).

---

## Terminal 3 — Autonomous Distillation Agent

This is the main script — runs the full pipeline end-to-end. Start this after the monitors are up.

### Production Launcher (tmux + caffeinate — recommended)

For long multi-trial runs use the production launcher. It automatically wraps itself in a tmux session (`distill-prod`) so the run survives terminal disconnects, and uses `caffeinate -s` to prevent the Mac from sleeping while on AC power:

```bash
./scripts/run_autonomous_production.sh
```

Detach at any time with `Ctrl-B D` and re-attach later with `tmux attach -t distill-prod`.

Requires `tmux`: `brew install tmux`

---

### Recommended Run (MLX backend — 2–5× faster on M3)

```bash
python -m distill.run_distillation_agent \
  --open \
  --backend mlx \
  --epochs 2 \
  --max_samples 2000 \
  --export all \
  --watchdog \
  --log_experiment
```

### PyTorch Backend (maximum compatibility)

```bash
python -m distill.run_distillation_agent \
  --open \
  --backend pytorch \
  --epochs 2 \
  --max_samples 2000 \
  --export gguf \
  --log_experiment \
  --watchdog
```

### Full Autonomous Run (all features, MLX)

```bash
python -m distill.run_distillation_agent \
  --open \
  --backend mlx \
  --epochs 2 \
  --max_samples 2000 \
  --export all \
  --compare_teacher \
  --benchmarks \
  --log_experiment \
  --watchdog \
  --seed 42
```

### Multi-Trial Hyperparameter Search

```bash
python -m distill.run_distillation_agent \
  --open \
  --backend mlx \
  --n_trials 3 \
  --epochs 2 \
  --export all \
  --log_experiment
```

### Using Config File

```bash
# Edit configs/agent_config.json first, then:
python -m distill.run_distillation_agent --config configs/agent_config.json
```

#### The Agent Pipeline Sequence

The agent internally runs these scripts in this order, automatically:

1. `experiment_log.py` — Reads history, prints past runs
2. `generate_synthetic_data.py` — (if `--synthetic_data`)
3. `distill_sft.py` — (if `--curriculum`)
4. Backend training script — `distill_mlx.py`, `distill_minillm.py`, or `distill_unsloth.py` depending on `--backend`
5. Export — GGUF / CoreML / MLX quantization depending on `--export`
6. `run_eval.py` — Validation perplexity + teacher comparison
7. `eval_quality.py` — Diversity, refusal rate, judge score
8. `run_benchmarks.py` — (if `--benchmarks`) WikiText-2 perplexity
9. `experiment_log.py` — Writes result + diagnostics to log

---

## Terminal 4 — Gradio Dashboard

Start this any time — before, during, or after training. It reads live metrics while the agent runs.

```bash
python -m distill.dashboard \
  --runs_dir . \
  --port 7860
```

Then open http://127.0.0.1:7860 in your browser.

### Dashboard Tabs and Data Sources

| Tab         | Populated by                                      |
|-------------|---------------------------------------------------|
| Plots       | `metrics.jsonl` (live during training)            |
| Pipeline    | `trainer_state.json` + GGUF files (after export)  |
| Thermal     | `thermal.log` (if thermal logger running)         |
| Evaluate    | Auto-discovers models from HF cache + output dirs |
| Quality     | `quality_metrics.json` (after `eval_quality.py`)  |
| Experiments | `experiment_log.jsonl` (after `--log_experiment`) |

---

## Verification

Scripts and configs verified as of 2026-03-08:

- ✓ All referenced scripts exist in `scripts/`
- ✓ Config files exist: `configs/watchdog_rules.json`, `configs/agent_config.json`
- ✓ All command-line arguments verified against current script `parse_args()`
- ✓ `--export all` supported (GGUF + CoreML + MLX in one run)
- ✓ `--backend mlx` recommended for M3 (2–5× faster than pytorch)

---

## Common Output Directories

Default output structure (MLX backend):

```
distill/
├── distilled-mlx/                  # Training output (MLX)
│   ├── distill_config.json         # Run configuration
│   ├── metrics.jsonl               # Training + eval loss curves
│   ├── mlx_student_weights.npz     # Trained LoRA weights
│   ├── mlx_q4/                     # 4-bit quantized MLX weights
│   ├── student-f16.gguf            # GGUF export (if --export gguf/all)
│   ├── distilled-mlx.mlpackage     # CoreML export (if --export coreml/all)
│   ├── quality_metrics.json        # Quality eval results
│   └── watchdog_suggestions.json   # Watchdog LR suggestions
├── thermal.log                     # Thermal monitoring log (optional)
├── experiment_log.jsonl            # Multi-run history
└── configs/
    ├── agent_config.json
    └── watchdog_rules.json
```

PyTorch backend uses `distilled-minillm/` and saves `pytorch_model.bin` / safetensors instead of `.npz`.

---

## Troubleshooting

### Watchdog Not Starting

- Ensure output directory exists: `mkdir -p ./distilled-mlx`
- Check config file syntax: `python -m json.tool configs/watchdog_rules.json`

### Thermal Agent / Monitor Fails

- Install mactop: `brew install context-labs/tap/mactop`
- Verify it works: `mactop --headless --format json --count 1`
- For fan control GUI: install [Macs Fan Control](https://crystalidea.com/macs-fan-control)

### Dashboard Shows No Models

- Check `--runs_dir` points to the parent directory containing training outputs
- Verify training has started and `metrics.jsonl` exists in the output dir

### Agent Fails to Start

- Add `--offline` if air-gapped
- Pre-download models: `python -m distill.cache_models --open`
- Check MLX is installed: `python -c "import mlx_lm; print('OK')"`
