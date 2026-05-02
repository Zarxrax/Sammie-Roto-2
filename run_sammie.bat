@echo off
setlocal
set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
start /b "" uvw run --no-sync launcher.py %*
exit