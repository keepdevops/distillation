#!/usr/bin/env bash
#
# Monitor CPU/GPU temperature and power on Apple Silicon M3 Max.
# Uses mactop (no sudo required).
#
# Usage:
#   ./scripts/monitor_cpu_gpu_temp.sh            # 3s interval
#   ./scripts/monitor_cpu_gpu_temp.sh 5          # 5s interval
#   ./scripts/monitor_cpu_gpu_temp.sh 3 thermal.log  # log to file

set -e

INTERVAL="${1:-3}"
LOGFILE="${2:-}"
INTERVAL_MS=$(( INTERVAL * 1000 ))

header="time                   CPU°C   GPU°C   SOC°C   CPU(W)  GPU(W)  Total(W)"
divider="------------------------------------------------------------------------"

echo "$header"
echo "$divider"
[ -n "$LOGFILE" ] && { echo "$header"; echo "$divider"; } >> "$LOGFILE"

while true; do
  JSON=$(mactop --headless --format json --count 1 2>/dev/null | \
         python3 -c "import sys,json; d=json.load(sys.stdin)[0]['soc_metrics']; \
           print(d['cpu_temp'], d['gpu_temp'], d['soc_temp'], \
                 d['cpu_power'], d['gpu_power'], d['total_power'])")
  read CPU_T GPU_T SOC_T CPU_W GPU_W TOT_W <<< "$JSON"
  TS=$(date "+%Y-%m-%d %H:%M:%S")
  LINE=$(printf "%s   %5.1f   %5.1f   %5.1f   %5.2f   %5.2f   %7.2f" \
    "$TS" "$CPU_T" "$GPU_T" "$SOC_T" "$CPU_W" "$GPU_W" "$TOT_W")
  echo "$LINE"
  [ -n "$LOGFILE" ] && echo "$LINE" >> "$LOGFILE"
  sleep "$INTERVAL"
done
