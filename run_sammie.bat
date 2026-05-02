@echo off
setlocal
set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uvw.exe"
start /b "" "%UV_EXE%" run --no-sync launcher.py %*
exit