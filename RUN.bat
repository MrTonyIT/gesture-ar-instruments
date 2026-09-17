@echo off
title GESTURE AR INSTRUMENTS - MrTonyIT
echo ========================================================
echo   KHOI DONG GESTURE AR INSTRUMENTS (AIR GUITAR & PIANO)
echo ========================================================
echo.
cd /d "d:\AImusic\gesture_ar_instruments"
".venv\Scripts\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Co loi xay ra khi chay ung dung!
    pause
)
