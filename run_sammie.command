#!/usr/bin/env bash

# Change to script directory
cd "$(dirname "$0")" || { echo "Failed to change to script directory"; exit 1; }

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run install_dependencies.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Run the application
python3 launcher.py "$@"