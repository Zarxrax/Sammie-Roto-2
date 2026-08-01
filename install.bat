@echo off
setlocal

:: Change directory to the script location
cd /d "%~dp0"

:: Define environment variables
set "UV_DIR=%~dp0.uv"
set "UV_EXE=%UV_DIR%\uv.exe"
::set "UV_PYTHON_INSTALL_DIR=%UV_DIR%\python"
::set "UV_CACHE_DIR=%UV_DIR%\uv_cache"
set "UV_VERSION=0.12.0"

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
    echo This drive's filesystem doesn't support the links uv needs ^(common on exFAT^) -- using local app data instead for Python/cache storage.
    set "UV_PYTHON_INSTALL_DIR=%LOCALAPPDATA%\Sammie-Roto-2\uv-python"
    set "UV_CACHE_DIR=%LOCALAPPDATA%\Sammie-Roto-2\uv-cache"
    :: Cache is now on a different drive than .venv -- hardlinks can't
    :: cross that boundary regardless of filesystem type, so force copies.
    set "UV_LINK_MODE=copy"
)

:: Install uv locally if missing
if not exist "%UV_EXE%" (
    echo Downloading uv to isolated folder...

    powershell -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%UV_DIR%'; irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex"

    if errorlevel 1 (
        echo Failed to install uv.
        pause
        exit /b 1
    )

    if not exist "%UV_EXE%" (
        echo uv.exe was not installed.
        pause
        exit /b 1
    )
)

:: Execute the install script through uv
echo Running installer...
"%UV_EXE%" run --no-project --with dulwich~=1.2 --python 3.12 python manage.py %*

:: Catch error from install script
set "MANAGE_EXIT=%ERRORLEVEL%"
if not "%MANAGE_EXIT%"=="0" (
    echo  Setup did not finish cleanly (exit code %MANAGE_EXIT%^).
)

pause