#!/usr/bin/env bash

# Always operate relative to this script's folder
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to cd to script directory"; exit 1; }

# Function to check Python version
check_python_version() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ python3 is not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.10"

    version_ge() {
        [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
    }

    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
        echo "❌ Python 3.10 or higher is required. Detected version: $PYTHON_VERSION"
        exit 1
    fi
	
	# Check if venv is available
	if ! python3 -m venv --help &> /dev/null; then
		echo "❌ Error: The 'venv' module is not available in this Python installation."
		echo "Try reinstalling Python from https://www.python.org/ or via your package manager."
		exit 1
	fi
	
	echo "✅ Detected compatible Python version: $PYTHON_VERSION"
}

# Check Python version
check_python_version

# Detect OS
OS_TYPE="$(uname)"

# Ask user what they want to do
echo
echo "❓ What would you like to do?"
select ACTION in "Full installation" "Download models only (Only needed if the models fail to download)" "Exit"; do
    case $ACTION in
        "Full installation")
            echo "Proceeding with full installation..."
            break
            ;;
        "Download models only (Only needed if the models fail to download)")
            echo "Downloading models..."
            if [ -d "venv" ]; then
                source venv/bin/activate
            fi
            python3 sammie/download_models.py
            echo "✅ Model download complete!"
            exit 0
            ;;
        "Exit")
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please choose 1, 2, or 3."
            ;;
    esac
done

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
echo "Upgrading pip..."
pip3 install --upgrade pip
echo "Installing wheel..."
pip3 install wheel

# Install PyTorch
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Detected macOS. Installing PyTorch..."
    pip3 install torch==2.9.1 torchvision
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Detected Linux."
	echo
    echo "❓ Which PyTorch build do you want to install?"
    select PT_VERSION in "CUDA 12.8 (For modern NVIDIA GPUs, RTX)" "CUDA 12.6 (For old NVIDIA GPUs, GXT)" "ROCm (Radeon)" "CPU"; do
        case $PT_VERSION in
            "CUDA 12.8 (For modern NVIDIA GPUs, RTX)")
                echo "Installing PyTorch with CUDA 12.8..."
                pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
                break
                ;;
            "CUDA 12.6 (For old NVIDIA GPUs, GXT)")
                echo "Installing PyTorch with CUDA 12.6..."
                pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
                break
                ;;
            "ROCm (Radeon)")
                echo "Installing PyTorch with ROCm 6.4..."
                pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/rocm6.4
                break
                ;;
            CPU)
                echo "Installing CPU-only version of PyTorch..."
                pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cpu
                break
                ;;
            *)
                echo "Invalid option. Please choose 1, 2, 3, or 4."
                ;;
        esac
    done
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Setting the application to executable
chmod +x run_sammie.command

# Install other dependencies
echo "Installing requirements..."
pip3 install -r requirements.txt

# Run model downloader
echo "Downloading models..."
python3 sammie/download_models.py

echo "✅ Setup complete!"