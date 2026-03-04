#!/bin/bash
set -e

# Install thermal_agent.py as macOS LaunchAgent
# Runs automatically on login, survives reboots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_NAME="com.distillation.thermal_agent.plist"
PLIST_PATH="$LAUNCH_AGENTS/$PLIST_NAME"

# Get Python path from conda environment
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_PATH="$CONDA_PREFIX/bin/python"
else
    PYTHON_PATH="$(which python3)"
fi

# Configuration
WATCH_DIR="${WATCH_DIR:-$PROJECT_DIR}"
THRESHOLD="${THRESHOLD:-85}"
INTERVAL="${INTERVAL:-30}"
LOG_FILE="$PROJECT_DIR/thermal_agent.jsonl"

echo "════════════════════════════════════════════════════════════════════"
echo "  Install Thermal Agent LaunchAgent"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Python:    $PYTHON_PATH"
echo "  Script:    $SCRIPT_DIR/thermal_agent.py"
echo "  Watch:     $WATCH_DIR"
echo "  Threshold: ${THRESHOLD}°C"
echo "  Interval:  ${INTERVAL}s"
echo "  Log:       $LOG_FILE"
echo ""

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS"

# Create plist file
cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.distillation.thermal_agent</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>$SCRIPT_DIR/thermal_agent.py</string>
        <string>--watch</string>
        <string>$WATCH_DIR</string>
        <string>--threshold</string>
        <string>$THRESHOLD</string>
        <string>--interval</string>
        <string>$INTERVAL</string>
        <string>--log</string>
        <string>$LOG_FILE</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/thermal_agent.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/thermal_agent.stderr.log</string>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>Nice</key>
    <integer>10</integer>
</dict>
</plist>
EOF

echo "✓ Created LaunchAgent plist: $PLIST_PATH"
echo ""

# Stop existing agent if running
if launchctl list | grep -q com.distillation.thermal_agent; then
    echo "Stopping existing thermal agent..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# Load agent
echo "Loading thermal agent..."
launchctl load "$PLIST_PATH"

# Verify it started
sleep 2
if launchctl list | grep -q com.distillation.thermal_agent; then
    echo ""
    echo "✓ Thermal agent installed and running!"
    echo ""
    echo "Status commands:"
    echo "  launchctl list | grep thermal_agent          # Check status"
    echo "  tail -f $LOG_FILE                            # View thermal log"
    echo "  tail -f $PROJECT_DIR/thermal_agent.stdout.log  # View stdout"
    echo ""
    echo "Management commands:"
    echo "  launchctl unload $PLIST_PATH    # Stop agent"
    echo "  launchctl load $PLIST_PATH      # Start agent"
    echo "  rm $PLIST_PATH && launchctl stop $PLIST_NAME  # Uninstall"
    echo ""
else
    echo ""
    echo "⚠ Thermal agent may not be running. Check logs:"
    echo "  tail -f $PROJECT_DIR/thermal_agent.stderr.log"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Thermal Agent Active"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "The thermal agent will:"
echo "  • Monitor system temperature continuously"
echo "  • Pause ALL jobs when temp exceeds ${THRESHOLD}°C"
echo "  • Resume jobs automatically when temp drops"
echo "  • Survive reboots (starts on login)"
echo "  • Protect all distillation jobs system-wide"
echo ""
