#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_DIR="$SCRIPT_DIR/.uv"
UV_EXE="$UV_DIR/uv"

# Set environment variables for isolation
export UV_INSTALL_DIR="$UV_DIR"
export UV_UNMANAGED_INSTALL=1
export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export UV_CACHE_DIR="$UV_DIR/uv_cache"

# Install uv locally if missing
if [ ! -f "$UV_EXE" ]; then
    echo "Downloading uv to isolated folder..."
    mkdir -p "$UV_DIR"
    # Use the official shell installer script
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Add the isolated folder to this session's PATH
export PATH="$UV_DIR:$PATH"

echo "Running installer..."
# Run the application with required dependencies
uv run --no-project --with "dulwich~=1.2" python manage.py