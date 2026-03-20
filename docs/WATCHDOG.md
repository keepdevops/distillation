# Autonomous Training Watchdog

ML-specific monitoring for long distillation runs. Detects loss plateau and divergence via deterministic rules. No LLM agent — just heartbeat checks and atomic writes. Production-safe on M3 Max.

**Thermal protection is separate:** Use `thermal_agent.py` for hardware temperature monitoring. The watchdog handles ML-specific signals (loss curves) only.

## Architecture

```
┌─────────────────────┐     poll      ┌──────────────────────────┐
│  training_watchdog  │ ────────────▶ │ trainer_state.json        │
│  (LaunchAgent or    │               │ (written by training loop) │
│   terminal)         │               └──────────────────────────┘
└─────────┬───────────┘
          │ writes
          ▼
┌─────────────────────┐     reads     ┌──────────────────────────┐
│ watchdog_suggestions│ ◀──────────── │ PauseFlagCallback         │
│ pause.flag          │               │ (in Trainer / --watchdog) │
└─────────────────────┘               └──────────────────────────┘
```

## Rules (configurable)

| Rule | Trigger | Action |
|------|---------|--------|
| **Plateau** | Last N loss deltas all < `max_delta` (default 0.001) | Writes `watchdog_suggestions.json` with scaled LR recommendation |
| **Divergence** | Recent avg loss > baseline avg × `threshold` (default 1.5×) | Writes `pause.flag` immediately → trainer saves and exits |

Minimum points (`min_points`) must be logged before either check activates.

## Quick Start

### 1. Enable pause.flag support in training

Add `--watchdog` to any training command:

```bash
# MLX backend (recommended)
python scripts/distill_mlx.py --open --watchdog --output_dir ./distilled-mlx

# PyTorch backend
python scripts/distill_minillm.py --open --watchdog --output_dir ./distilled-minillm
```

### 2. Run watchdog in a separate terminal

**Python:**

```bash
# Continuous (every 60s)
python scripts/training_watchdog.py ./distilled-mlx --interval 60

# With config file
python scripts/training_watchdog.py ./distilled-mlx --interval 60 \
  --config configs/watchdog_rules.json

# One-off check
python scripts/training_watchdog.py ./distilled-mlx --once
```

**C++** (no Python, lighter):

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build .
./watchdog ../../distilled-mlx --interval 60 --config ../../configs/watchdog_rules.json
```

### 3. LaunchAgent (survives reboot)

```bash
# Edit paths in scripts/launch_agent/com.caribou.distill-watchdog.plist
cp scripts/launch_agent/com.caribou.distill-watchdog.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
```

## Config

`configs/watchdog_rules.json`:

```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "min_points": 5,
    "lr_scale": 0.8
  },
  "divergence": {
    "window": 3,
    "threshold": 1.5,
    "baseline_window": 5,
    "min_points": 8
  }
}
```

### Plateau parameters

| Key | Default | Description |
|-----|---------|-------------|
| `window` | 3 | Number of consecutive loss deltas to check |
| `max_delta` | 0.001 | If all N deltas are below this → plateau |
| `min_points` | 5 | Minimum loss entries before plateau check starts |
| `lr_scale` | 0.8 | Suggested LR multiplier written to `watchdog_suggestions.json` |

### Divergence parameters

| Key | Default | Description |
|-----|---------|-------------|
| `window` | 3 | Recent average = mean of last N losses |
| `threshold` | 1.5 | Pause if recent avg > baseline avg × this |
| `baseline_window` | 5 | Baseline = mean of first N losses |
| `min_points` | 8 | Minimum points before divergence check starts |

## Outputs

- **`watchdog_suggestions.json`** — Proposed params for next run (e.g. `next_lr_scale: 0.8`) written on plateau detection
- **`pause.flag`** — Signals trainer to save checkpoint and exit cleanly (written on divergence)

## Thermal Protection

Thermal monitoring (CPU/GPU/SoC temperature) is handled by the **separate** `thermal_agent.py`, not this watchdog. The two cooperate via `pause.flag` metadata:

```bash
# Run thermal agent alongside watchdog for complete protection
python scripts/thermal_agent.py --watch . --threshold 85   # terminal 1
python scripts/training_watchdog.py ./distilled-mlx        # terminal 2
python scripts/distill_mlx.py --open --watchdog            # terminal 3
```

See [THERMAL_AGENT.md](../THERMAL_AGENT.md) for thermal agent setup.

## Integration with Other Backends

The watchdog reads `trainer_state.json` (standard HuggingFace Trainer format). This is written automatically by:
- `distill_mlx.py` — writes after each step
- `distill_minillm.py` — written by HF Trainer
- `distill_unsloth.py` — written by HF Trainer

Point the watchdog at whatever `--output_dir` you pass to the training script.
