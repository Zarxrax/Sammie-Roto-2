@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:: Check if running directly from an unextracted ZIP or temporary directory
set "HERE=%~dp0"
set "IN_TEMP=0"
if /i not "!HERE:.zip\=!"=="!HERE!" set "IN_TEMP=1"
if /i not "!HERE:\AppData\Local\Temp\=!"=="!HERE!" set "IN_TEMP=1"
if defined TEMP if /i not "!HERE:%TEMP%\=!"=="!HERE!" set "IN_TEMP=1"
if "!IN_TEMP!"=="1" (
    echo ======================================================================
    echo ERROR: You appear to be running Sammie-Roto directly from inside a ZIP file!
    echo.
    echo Please extract the downloaded ZIP file before running the program.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uvw.exe"
if not exist "%UV_EXE%" set "UV_EXE=%UV_DIR%\uv.exe"

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

:: 1. Check if program is not installed yet
if not exist "%VENV_DIR%" (
    echo ======================================================================
    echo Sammie-Roto is not installed yet.
    echo ======================================================================
    set /p "RUN_INSTALL=Would you like to run the installer now? (Y/n): "
    if /i "!RUN_INSTALL!"=="n" (
        exit /b 0
    )
    call "%~dp0install.bat" %*
    exit /b
)

:: 2. Check if virtual environment is valid (e.g. not moved or broken trampoline)
set "NEED_REPAIR=0"
if not exist "%VENV_PY%" (
    set "NEED_REPAIR=1"
) else (
    "%VENV_PY%" -c "exit(0)" >nul 2>&1
    if errorlevel 1 set "NEED_REPAIR=1"
)

if "!NEED_REPAIR!"=="1" (
    echo ======================================================================
    echo The virtual environment appears to be broken or was moved.
    echo ======================================================================
    set /p "RUN_REPAIR=Would you like to run the installer to repair it? (Y/n): "
    if /i "!RUN_REPAIR!"=="n" (
        exit /b 0
    )
    call "%~dp0install.bat" %*
    exit /b
)

:: Pass storage locations if local .uv folder exists
if exist "%UV_DIR%\python" set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"
if exist "%UV_DIR%\uv_cache" set "UV_CACHE_DIR=%UV_DIR%\uv_cache"

:: Launch the application
start /b "" "%UV_EXE%" run --no-sync launcher.py %*
exit