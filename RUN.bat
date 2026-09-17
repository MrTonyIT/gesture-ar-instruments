@echo off
setlocal
title GESTURE AR INSTRUMENTS - MrTonyIT
echo ========================================================
echo   KHOI DONG GESTURE AR INSTRUMENTS (AIR GUITAR & PIANO)
echo ========================================================
echo.
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python environment not found!
    echo Please install Python 3.10+ or create a virtual environment in .venv.
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] An error occurred while running the application (Exit code %ERRORLEVEL%).
    pause
)
