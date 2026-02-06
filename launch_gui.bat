@echo off
title YOLO Training Pipeline - DEPRECATED LAUNCHER
color 0C
echo.
echo ========================================
echo   WARNING: DEPRECATED LAUNCHER
echo ========================================
echo.
echo This launcher uses SYSTEM PYTHON which has DLL errors!
echo.
echo PyTorch 2.10.0+cu128 has a known DLL initialization bug.
echo.
echo ========================================
echo   PLEASE USE: launch_gui_venv.bat
echo ========================================
echo.
echo The virtual environment has PyTorch 2.1.0+cu121 which works correctly.
echo.
echo Press any key to launch anyway (NOT RECOMMENDED)
echo Or close this window and use launch_gui_venv.bat instead
pause
echo.

REM Deactivate venv if active
if defined VIRTUAL_ENV (
    echo Deactivating virtual environment...
    call deactivate 2>nul
)

REM Clear Python path to avoid venv
set PYTHONPATH=

echo Starting with system Python (expect DLL errors)...
echo.
C:\Users\andyk\AppData\Local\Programs\Python\Python310\python.exe launch_gui_system.py
echo.
echo Application closed.
pause

