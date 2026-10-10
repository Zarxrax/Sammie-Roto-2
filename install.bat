@echo off
setlocal EnableDelayedExpansion

:: Change directory to the script location
cd /d "%~dp0"

:: Check if running directly from an unextracted ZIP or temporary directory
set "HERE=%~dp0"
set "IN_TEMP=0"
if /i not "!HERE:.zip\=!"=="!HERE!" set "IN_TEMP=1"
if /i not "!HERE:\AppData\Local\Temp\=!"=="!HERE!" set "IN_TEMP=1"
if defined TEMP if /i not "!HERE:%TEMP%\=!"=="!HERE!" set "IN_TEMP=1"
if "!IN_TEMP!"=="1" (
    echo ======================================================================
    echo ERROR: You appear to be running this script directly from inside a ZIP file!
    echo.
    echo Windows cannot install the program properly from a temporary folder.
    echo Please extract the downloaded ZIP file first:
    echo   1. Right-click the downloaded ZIP file.
    echo   2. Select "Extract All..." and extract to a permanent folder.
    echo   3. Open the extracted folder and run install.bat from there.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

:: Define environment variables
set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_VERSION=0.12.5"
if not defined UV_HTTP_TIMEOUT set "UV_HTTP_TIMEOUT=100"

if not exist "%UV_DIR%" mkdir "%UV_DIR%"

:: uv needs junctions/hardlinks for Python installs and its package cache;
:: exFAT doesn't support them and fail with "incorrect function (os error 1)". 
:: Test with a junction and fall back only if needed.
set "FS_TEST_DIR=%UV_DIR%\_fs_test"
set "FS_TEST_LINK=%UV_DIR%\_fs_test_link"
if exist "%FS_TEST_LINK%" rmdir "%FS_TEST_LINK%" >nul 2>&1
if exist "%FS_TEST_DIR%" rmdir "%FS_TEST_DIR%" >nul 2>&1
mkdir "%FS_TEST_DIR%"
mklink /J "%FS_TEST_LINK%" "%FS_TEST_DIR%" >nul 2>&1

if exist "%FS_TEST_LINK%" (
    set "FS_SUPPORTS_LINKS=1"
    rmdir "%FS_TEST_LINK%"
) else (
    set "FS_SUPPORTS_LINKS=0"
)
rmdir "%FS_TEST_DIR%"

if "%FS_SUPPORTS_LINKS%"=="1" (
    set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"
    set "UV_CACHE_DIR=%UV_DIR%\uv_cache"
) else (
    echo This drive's filesystem doesn't support the links uv needs ^(common on exFAT/FAT32^) -- using local app data instead for Python/cache storage.
    set "UV_PYTHON_INSTALL_DIR=%LOCALAPPDATA%\Sammie-Roto-2\uv-python"
    set "UV_CACHE_DIR=%LOCALAPPDATA%\Sammie-Roto-2\uv-cache"
    :: Cache is now on a different drive than .venv -- hardlinks can't
    :: cross that boundary regardless of filesystem type, so force copies.
    set "UV_LINK_MODE=copy"
)

:: Install uv locally if missing
if not exist "%UV_EXE%" (
    echo Downloading uv ^(package manager used for setup^)...

    powershell -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%UV_DIR%'; irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex" > "%UV_DIR%\uv_install.log" 2>&1

    if errorlevel 1 (
        echo Failed to install uv. Details:
        type "%UV_DIR%\uv_install.log"
        pause
        exit /b 1
    )

    if not exist "%UV_EXE%" (
        echo uv.exe was not installed. Details:
        type "%UV_DIR%\uv_install.log"
        pause
        exit /b 1
    )

    del "%UV_DIR%\uv_install.log" >nul 2>&1
    echo uv downloaded.
    echo.
    echo Preparing Python and the installer. Please wait...
)

:: One-time bootstrap venv for running manage.py itself (needs dulwich for git operations).
set "BOOTSTRAP_DIR=%UV_DIR%\bootstrap"
set "BOOTSTRAP_PY=%BOOTSTRAP_DIR%\Scripts\python.exe"

:: Validate existing bootstrap environment (detects moved directories or broken trampolines)
set "BOOTSTRAP_VALID=0"
if exist "%BOOTSTRAP_PY%" (
    "%BOOTSTRAP_PY%" -c "import dulwich" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_VALID=1"
)

if "%BOOTSTRAP_VALID%"=="0" (
    if exist "%BOOTSTRAP_DIR%" (
        echo Existing installer environment is invalid or was moved. Recreating...
        rmdir /s /q "%BOOTSTRAP_DIR%" >nul 2>&1
    )

    :: Clean up any dangling Python junctions (e.g. from a moved or packaged folder)
    if exist "%UV_DIR%\python" (
        powershell -ExecutionPolicy Bypass -NoProfile -Command "Get-ChildItem -Path '%UV_DIR%\python' -Force -ErrorAction SilentlyContinue | Where-Object { $_.LinkType -eq 'Junction' -and -not (Test-Path $_.Target) } | Remove-Item -Force" >nul 2>&1
    )

    echo Setting up installer environment...
    "%UV_EXE%" venv --python 3.12 --python-preference only-managed "%BOOTSTRAP_DIR%"
    if errorlevel 1 (
        echo Failed to create installer environment.
        pause
        exit /b 1
    )

    "%UV_EXE%" pip install --python "%BOOTSTRAP_PY%" "dulwich~=1.2"
    if errorlevel 1 (
        echo Failed to install installer dependencies.
        rmdir /s /q "%BOOTSTRAP_DIR%" >nul 2>&1
        pause
        exit /b 1
    )
)

:: Execute the install script in an in-memory block to guard against self-modification.
:: If manage.py pulls git commits that rewrite install.bat on disk, cmd.exe keeps
:: executing from the buffered block in memory rather than reading corrupted offsets.
(
    echo Running installer...
    "%BOOTSTRAP_PY%" manage.py %*
    set "MANAGE_EXIT=!ERRORLEVEL!"
    if not "!MANAGE_EXIT!"=="0" (
        echo  Setup did not finish cleanly ^(exit code !MANAGE_EXIT!^).
    )
    pause
    exit /b !MANAGE_EXIT!
)