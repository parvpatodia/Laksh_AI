#!/usr/bin/env bash

# =============================================================================
# Schedule Setup for Background Agent Runner
# =============================================================================
#
# This script sets up scheduled execution of the background agent.
# Supports macOS (launchd) and Linux (cron).
#
# USAGE:
#   ./setup-schedule.sh <project-path>
#
# This will:
#   1. Detect your OS
#   2. Set up scheduled tasks for the background runner
#   3. Configure a desktop notification for end-of-day push prompt
#
# =============================================================================

set -euo pipefail

PROJECT_PATH="${1:?Usage: $0 <project-path>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/background-runner.sh"

# Resolve absolute project path
PROJECT_PATH="$(cd "$PROJECT_PATH" && pwd)"

echo "Setting up background agent schedule for: $PROJECT_PATH"
echo "Using runner at: $RUNNER"
echo ""

# Make runner executable
chmod +x "$RUNNER"

# --- Detect OS ---
OS="$(uname -s)"

setup_macos() {
    echo "Detected macOS. Setting up launchd agents..."

    PLIST_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$PLIST_DIR"

    # Sanitize project path for use as label
    LABEL_SUFFIX=$(echo "$PROJECT_PATH" | sed 's/[^a-zA-Z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//')

    # --- End of Day (5:30 PM weekdays) ---
    cat > "${PLIST_DIR}/com.agent.endofday.${LABEL_SUFFIX}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.agent.endofday.${LABEL_SUFFIX}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${RUNNER}</string>
        <string>${PROJECT_PATH}</string>
        <string>end-of-day</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <dict>
            <key>Hour</key><integer>17</integer>
            <key>Minute</key><integer>30</integer>
            <key>Weekday</key><integer>1</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>17</integer>
            <key>Minute</key><integer>30</integer>
            <key>Weekday</key><integer>2</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>17</integer>
            <key>Minute</key><integer>30</integer>
            <key>Weekday</key><integer>3</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>17</integer>
            <key>Minute</key><integer>30</integer>
            <key>Weekday</key><integer>4</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>17</integer>
            <key>Minute</key><integer>30</integer>
            <key>Weekday</key><integer>5</integer>
        </dict>
    </array>
    <key>StandardOutPath</key>
    <string>${PROJECT_PATH}/.agent-logs/endofday-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_PATH}/.agent-logs/endofday-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

    # --- Health Check (every 3 hours during work hours) ---
    cat > "${PLIST_DIR}/com.agent.healthcheck.${LABEL_SUFFIX}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.agent.healthcheck.${LABEL_SUFFIX}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${RUNNER}</string>
        <string>${PROJECT_PATH}</string>
        <string>health-check</string>
    </array>
    <key>StartInterval</key>
    <integer>10800</integer>
    <key>StandardOutPath</key>
    <string>${PROJECT_PATH}/.agent-logs/healthcheck-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_PATH}/.agent-logs/healthcheck-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

    # --- Morning Review (7 AM weekdays) ---
    cat > "${PLIST_DIR}/com.agent.review.${LABEL_SUFFIX}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.agent.review.${LABEL_SUFFIX}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${RUNNER}</string>
        <string>${PROJECT_PATH}</string>
        <string>review</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <dict>
            <key>Hour</key><integer>7</integer>
            <key>Minute</key><integer>0</integer>
            <key>Weekday</key><integer>1</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>7</integer>
            <key>Minute</key><integer>0</integer>
            <key>Weekday</key><integer>2</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>7</integer>
            <key>Minute</key><integer>0</integer>
            <key>Weekday</key><integer>3</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>7</integer>
            <key>Minute</key><integer>0</integer>
            <key>Weekday</key><integer>4</integer>
        </dict>
        <dict>
            <key>Hour</key><integer>7</integer>
            <key>Minute</key><integer>0</integer>
            <key>Weekday</key><integer>5</integer>
        </dict>
    </array>
    <key>StandardOutPath</key>
    <string>${PROJECT_PATH}/.agent-logs/review-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_PATH}/.agent-logs/review-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

    # Load the agents
    for plist in "${PLIST_DIR}"/com.agent.*.${LABEL_SUFFIX}.plist; do
        launchctl load "$plist" 2>/dev/null || true
        echo "  Loaded: $(basename "$plist")"
    done

    echo ""
    echo "macOS setup complete. Scheduled tasks:"
    echo "  - Health check: every 3 hours"
    echo "  - Code review: 7:00 AM weekdays"
    echo "  - End-of-day summary: 5:30 PM weekdays"
    echo ""
    echo "To add a desktop notification when end-of-day runs, add this to"
    echo "the end of background-runner.sh's end-of-day case:"
    echo ""
    echo '  osascript -e "display notification \"End-of-day summary ready. Check END_OF_DAY.md\" with title \"Agent: Ready to Push?\" sound name \"Glass\""'
    echo ""
    echo "To unload all agents:"
    echo "  for f in ${PLIST_DIR}/com.agent.*.${LABEL_SUFFIX}.plist; do launchctl unload \"\$f\"; done"
}

setup_linux() {
    echo "Detected Linux. Setting up cron jobs..."

    # Create a temporary cron file
    CRON_TMP=$(mktemp)

    # Export existing crontab (if any)
    crontab -l > "$CRON_TMP" 2>/dev/null || true

    # Remove any existing agent entries for this project
    grep -v "$RUNNER.*$PROJECT_PATH" "$CRON_TMP" > "${CRON_TMP}.clean" || true
    mv "${CRON_TMP}.clean" "$CRON_TMP"

    # Add new entries
    cat >> "$CRON_TMP" <<EOF

# --- Background Agent: $(basename "$PROJECT_PATH") ---
# Health check every 3 hours during work hours (9-18) on weekdays
0 9,12,15,18 * * 1-5  ${RUNNER} ${PROJECT_PATH} health-check
# Code review at 7 AM on weekdays
0 7 * * 1-5  ${RUNNER} ${PROJECT_PATH} review
# Research at 2 AM daily
0 2 * * *  ${RUNNER} ${PROJECT_PATH} research
# Suggestions every Monday at 8 AM
0 8 * * 1  ${RUNNER} ${PROJECT_PATH} suggest
# End-of-day summary at 5:30 PM on weekdays
30 17 * * 1-5  ${RUNNER} ${PROJECT_PATH} end-of-day
EOF

    crontab "$CRON_TMP"
    rm "$CRON_TMP"

    echo ""
    echo "Linux cron setup complete. Scheduled tasks:"
    echo "  - Health check: 9 AM, 12 PM, 3 PM, 6 PM weekdays"
    echo "  - Code review: 7:00 AM weekdays"
    echo "  - Research: 2:00 AM daily"
    echo "  - Suggestions: 8:00 AM Mondays"
    echo "  - End-of-day summary: 5:30 PM weekdays"
    echo ""
    echo "View with: crontab -l"
    echo "Edit with: crontab -e"
}

# --- Main ---

mkdir -p "${PROJECT_PATH}/.agent-logs"

case "$OS" in
    Darwin)
        setup_macos
        ;;
    Linux)
        setup_linux
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Manually set up cron jobs using background-runner.sh"
        exit 1
        ;;
esac

echo ""
echo "=== IMPORTANT ==="
echo ""
echo "Before this works, make sure:"
echo "  1. Claude Code is installed:  npm install -g @anthropic-ai/claude-code"
echo "  2. Claude Code is authenticated (run 'claude' once manually to set up)"
echo "  3. CLAUDE.md exists in $PROJECT_PATH (copy agent-rules.md there)"
echo "  4. GOALS.md exists in $PROJECT_PATH (fill out the template)"
echo ""
echo "Test manually first:"
echo "  ${RUNNER} ${PROJECT_PATH} health-check"
echo ""
echo "Logs will be written to: ${PROJECT_PATH}/.agent-logs/"
