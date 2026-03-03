# LaunchAgent for Distill Watchdog

Run the training watchdog as a LaunchAgent so it survives reboot and runs independently.

## Setup

1. **Edit the plist** to match your paths:
   - `ProgramArguments[0]`: path to your conda Python (e.g. `which python` in distill env)
   - `ProgramArguments[2]`: your training output_dir (where trainer_state.json lives)
   - `WorkingDirectory`: your distill project root

2. **Install:**
   ```bash
   cp com.caribou.distill-watchdog.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
   ```

3. **Or run manually** (same as LaunchAgent, but in terminal):
   ```bash
   python scripts/training_watchdog.py ./distilled-minillm --interval 60
   ```

## Usage

- **Start training first:** `python scripts/distill_minillm.py --output_dir ./distilled-minillm`
- **Then** load the LaunchAgent (or run watchdog in another terminal)
- Watchdog polls `trainer_state.json` every 60s; on plateau writes `watchdog_suggestions.json`
- Add `PauseFlagCallback` to your trainer for `pause.flag` support (thermal/emergency stop)

## Unload

```bash
launchctl unload ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
```
