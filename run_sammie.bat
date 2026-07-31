@echo off
setlocal
cd /d "%~dp0"

set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uvw.exe"

start /b "" "%UV_EXE%" run --no-sync launcher.py %*
exit