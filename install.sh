#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

UV_DIR="$SCRIPT_DIR/.uv"
UV_EXE="$UV_DIR/uv"
UV_VERSION="0.12.5"

mkdir -p "$UV_DIR"

# uv needs symlink/hardlink support for its Python installs and package cache; 
# exFAT doesn't support them and fail with "Operation not permitted (os error 1)". 
# Test directly and fall back only if needed.
FS_TEST_FILE="$UV_DIR/.fs_test_$$"
FS_TEST_LINK="$UV_DIR/.fs_test_link_$$"
: > "$FS_TEST_FILE" 2>/dev/null
if ln -s "$FS_TEST_FILE" "$FS_TEST_LINK" 2>/dev/null; then
    FS_SUPPORTS_LINKS=1
else
    FS_SUPPORTS_LINKS=0
fi
rm -f "$FS_TEST_FILE" "$FS_TEST_LINK"

if [ "$FS_SUPPORTS_LINKS" -eq 1 ]; then
    export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
    export UV_CACHE_DIR="$UV_DIR/uv_cache"
else
    echo "This drive's filesystem doesn't support symlinks (common on exFAT) -- using ~/.cache instead for Python/cache storage."
    export UV_PYTHON_INSTALL_DIR="$HOME/.cache/sammie-roto-2/uv-python"
    export UV_CACHE_DIR="$HOME/.cache/sammie-roto-2/uv-cache"
    # Cache is now on a different filesystem than .venv -- hardlinks can't
    # cross that boundary regardless of filesystem type, so force copies.
    export UV_LINK_MODE=copy
fi

# Install uv locally if missing
if [ ! -f "$UV_EXE" ]; then
    echo "Downloading uv to isolated folder..."
    # Use the official shell installer script
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | UV_INSTALL_DIR="$UV_DIR" sh

    if [ $? -ne 0 ]; then
        echo "Failed to install uv."
        exit 1
    fi

    if [ ! -f "$UV_EXE" ]; then
        echo "uv was not installed."
        exit 1
    fi
fi

# One-time bootstrap venv for running manage.py itself (needs dulwich for git operations).
BOOTSTRAP_DIR="$UV_DIR/bootstrap"
BOOTSTRAP_PY="$BOOTSTRAP_DIR/bin/python"
if [ ! -f "$BOOTSTRAP_PY" ]; then
    echo "Setting up installer environment..."
    "$UV_EXE" venv --python 3.12 "$BOOTSTRAP_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create installer environment."
        exit 1
    fi

    "$UV_EXE" pip install --python "$BOOTSTRAP_PY" "dulwich~=1.2"
    if [ $? -ne 0 ]; then
        echo "Failed to install installer dependencies."
        rm -rf "$BOOTSTRAP_DIR"
        exit 1
    fi
fi

echo "Running installer..."
# Run the application with required dependencies
"$BOOTSTRAP_PY" manage.py "$@"

# Catch error from install script
MANAGE_EXIT=$?
if [ "$MANAGE_EXIT" -ne 0 ]; then
    echo "Setup did not finish cleanly (exit code $MANAGE_EXIT)."
fi

# Only pause if stdin is a real terminal
if [ -t 0 ]; then
    read -rp "Press Enter to close..." _
fi
 
exit $MANAGE_EXIT
