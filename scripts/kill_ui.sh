#!/bin/bash
pids=$(pgrep -f "launch_ui.py")
if [ -z "$pids" ]; then
    echo "No launch_ui.py processes found."
else
    echo "Killing PIDs: $pids"
    kill $pids
fi
