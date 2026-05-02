#!/usr/bin/env bash

# Move to the directory where this script is located
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
UV_DIR="$SCRIPT_DIR/.uv"

export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export UV_CACHE_DIR="$UV_DIR/uv_cache"
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_QPA_PLATFORM="wayland;xcb"

"$UV_DIR/uv" run --no-sync launcher.py "$@"