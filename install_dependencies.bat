@echo off
pushd %~dp0

echo Select version to install:
echo 1. CUDA 12.8 (For modern NVIDIA GPUs, RTX)
echo 2. CUDA 12.6 (For old NVIDIA GPUs, GTX)
echo 3. CPU (Smaller download)
echo 4. Download models only (Only needed if the models fail to download)
echo 5. Exit

choice /C 12345 /N /M "Enter your choice (1, 2, 3, 4 or 5):"

if errorlevel 5 (
    echo Exiting...
    exit /b 0
) else if errorlevel 4 (
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
) else if errorlevel 3 (
    .\python-3.12.8-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install wheel --no-warn-script-location
    echo Uninstalling existing Pytorch if found
    .\python-3.12.8-embed-amd64\python.exe -m pip uninstall -y torch torchvision
    echo Installing CPU version of PyTorch
    .\python-3.12.8-embed-amd64\python.exe -m pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install -r .\requirements.txt --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
) else if errorlevel 2 (
    .\python-3.12.8-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install wheel --no-warn-script-location
    echo Uninstalling existing Pytorch if found
    .\python-3.12.8-embed-amd64\python.exe -m pip uninstall -y torch torchvision
    echo Installing CUDA 12.6 version of PyTorch
    .\python-3.12.8-embed-amd64\python.exe -m pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126 --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install -r .\requirements.txt --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
) else (
    .\python-3.12.8-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install wheel --no-warn-script-location
    echo Uninstalling existing Pytorch if found
    .\python-3.12.8-embed-amd64\python.exe -m pip uninstall -y torch torchvision
    echo Installing CUDA 12.8 version of PyTorch
    .\python-3.12.8-embed-amd64\python.exe -m pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe -m pip install -r .\requirements.txt --no-warn-script-location
    .\python-3.12.8-embed-amd64\python.exe .\sammie\download_models.py
)

echo Completed.
Pause