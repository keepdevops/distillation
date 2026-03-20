# LaunchAgent for Distill Watchdog

Run the training watchdog as a LaunchAgent so it survives reboot and runs independently.

## Setup

1. **Edit the plist** to match your paths:
   - `ProgramArguments[0]`: path to your conda Python (e.g. `which python` in distill env)
   - `ProgramArguments[2]`: your training output_dir (where `trainer_state.json` lives)
   - `WorkingDirectory`: your distill project root

2. **Install:**
   ```bash
   cp com.caribou.distill-watchdog.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
   ```

3. **Or run manually** (same as LaunchAgent, but in terminal):
   ```bash
   # MLX backend output dir (recommended)
   python scripts/training_watchdog.py ./distilled-mlx --interval 60

   # PyTorch backend output dir
   python scripts/training_watchdog.py ./distilled-minillm --interval 60
   ```

## Usage

- **Start training first** with `--watchdog` flag:
  ```bash
  python scripts/distill_mlx.py --open --watchdog --output_dir ./distilled-mlx
  # or
  python scripts/distill_minillm.py --open --watchdog --output_dir ./distilled-minillm
  ```
- **Then** load the LaunchAgent (or run watchdog in another terminal)
- Watchdog polls `trainer_state.json` every 60s
  - On **plateau**: writes `watchdog_suggestions.json` with scaled LR recommendation
  - On **divergence**: writes `pause.flag` → trainer saves and exits gracefully
- The `--watchdog` flag on the training script enables `PauseFlagCallback` automatically — no manual setup needed

## Unload

```bash
launchctl unload ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
```

## Notes

- For **thermal protection** (hardware temperature monitoring), use `thermal_agent.py` separately — it also writes `pause.flag` and cooperates with the watchdog without conflicts.
- See [WATCHDOG.md](../../docs/WATCHDOG.md) for full configuration reference.
