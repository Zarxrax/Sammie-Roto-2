@echo off
setlocal
cd /d "%~dp0"

:: Check if running directly from an unextracted ZIP or temporary directory
echo "%~dp0" | findstr /I /C:".zip\\" /C:"\AppData\Local\Temp\" /C:"\Temp\Temp" /C:"\Temp\7z" /C:"\Temp\Rar$" >nul
if not errorlevel 1 (
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

start /b "" "%UV_EXE%" run --no-sync launcher.py %*
exit