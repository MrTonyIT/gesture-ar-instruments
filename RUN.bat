@echo off
setlocal
title GESTURE AR INSTRUMENTS - MrTonyIT
echo ========================================================
echo   KHOI DONG GESTURE AR INSTRUMENTS (AIR GUITAR ^& PIANO)
echo ========================================================
echo.
cd /d "%~dp0"

set "PYTHON_EXE="
if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    if exist venv\Scripts\python.exe (
        set "PYTHON_EXE=venv\Scripts\python.exe"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python environment not found!
    echo Please install Python 3.10 or 3.11, or create a virtual environment in .venv.
    if not defined CI pause
    exit /b 1
)

:: Validate Python version contract: >=3.10,<3.12 (Python 3.10 or 3.11)
"%PYTHON_EXE%" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Incompatible Python interpreter!
    echo Gesture AR Instruments requires Python 3.10 or 3.11 [contract: ^>=3.10, ^<3.12].
    "%PYTHON_EXE%" -c "import sys; print('Detected Python version:', sys.version.split()[0])" 2>nul
    echo Please install or activate a compatible Python 3.10 or 3.11 environment.
    if not defined CI pause
    exit /b 1
)

"%PYTHON_EXE%" main.py %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if %APP_EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] An error occurred while running the application [Exit code %APP_EXIT_CODE%].
    if not defined CI pause
    exit /b %APP_EXIT_CODE%
)
exit /b 0
