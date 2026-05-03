@echo off
setlocal

:: Define the local folder
set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"
set "UV_CACHE_DIR=%UV_DIR%\uv_cache"

:: Install uv locally if missing
if not exist "%UV_EXE%" (
    echo Downloading uv to isolated folder...
    if not exist "%UV_DIR%" mkdir "%UV_DIR%"
    powershell -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%UV_DIR%'; irm https://astral.sh/uv/install.ps1 | iex"
)

:: Add the isolated folder to this session's PATH
set "PATH=%UV_DIR%;%PATH%"

echo Running installer...
uv run --no-project --with dulwich~=1.2 --python 3.12 python manage.py

pause