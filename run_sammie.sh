#!/usr/bin/env bash

# Move to the directory where this script is located
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
UV_DIR="$SCRIPT_DIR/.uv"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

# Show an error even when launched without a terminal
# (.desktop file with Terminal=false, or the macOS .app launcher)
notify_error() {
    local msg="$1"
    echo "$msg" >&2
    if [ "$(uname)" = "Darwin" ]; then
        osascript -e "display dialog \"$msg\" buttons {\"OK\"} default button \"OK\" with icon caution" >/dev/null 2>&1
    elif command -v zenity >/dev/null 2>&1; then
        zenity --error --text="$msg" >/dev/null 2>&1
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --error "$msg" >/dev/null 2>&1
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "Sammie-Roto-2" "$msg"
    fi
}

# 1. Check if program is not installed yet
if [ ! -d "$VENV_DIR" ] || [ ! -f "$UV_DIR/uv" ]; then
    if [ ! -t 0 ]; then
        notify_error "Sammie-Roto is not installed yet. Please open a terminal in the Sammie-Roto-2 folder and run ./install.sh"
        exit 1
    fi
    echo "======================================================================"
    echo "Sammie-Roto is not installed yet."
    echo "======================================================================"
    read -rp "Would you like to run the installer now? (Y/n): " RUN_INSTALL
    if [[ "$RUN_INSTALL" =~ ^[Nn] ]]; then
        exit 0
    fi
    exec bash ./install.sh "$@"
fi

# 2. Check if virtual environment is valid
if [ ! -f "$VENV_PY" ] || ! "$VENV_PY" -c "exit(0)" >/dev/null 2>&1; then
    if [ ! -t 0 ]; then
        notify_error "The Sammie-Roto installation appears to be broken or was moved. Please open a terminal in the Sammie-Roto-2 folder and run ./install.sh, then choose Reinstall/Repair"
        exit 1
    fi
    echo "======================================================================"
    echo "The virtual environment appears to be broken or was moved."
    echo "======================================================================"
    read -rp "Would you like to run the installer to repair it? (Y/n): " RUN_REPAIR
    if [[ "$RUN_REPAIR" =~ ^[Nn] ]]; then
        exit 0
    fi
    exec bash ./install.sh "$@"
fi

export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export UV_CACHE_DIR="$UV_DIR/uv_cache"

"$UV_DIR/uv" run --no-sync launcher.py "$@"