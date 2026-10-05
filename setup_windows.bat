@echo off
rem =============================================================================
rem AirDraw AI: AI-Powered Air Drawing & Gesture Interaction Platform
rem Automated Environment Setup for Microsoft Windows
rem =============================================================================
setlocal enabledelayedexpansion

echo ============================================================
echo  Setting up AirDraw AI Environment on Windows...
echo ============================================================

set FOUND=

py -3.12 -c "1" >nul 2>&1
if not errorlevel 1 (
    set FOUND=3.12
    goto found
)

py -3.11 -c "1" >nul 2>&1
if not errorlevel 1 (
    set FOUND=3.11
    goto found
)

py -3.10 -c "1" >nul 2>&1
if not errorlevel 1 (
    set FOUND=3.10
    goto found
)

py -3.9 -c "1" >nul 2>&1
if not errorlevel 1 (
    set FOUND=3.9
    goto found
)

echo.
echo ============================================================
echo  Error: Compatible Python not found. Need 3.9, 3.10, 3.11, or 3.12.
echo.
echo  Please install Python 3.12 from:
echo    https://www.python.org/downloads/
echo  Ensure "Add python.exe to PATH" is checked during setup.
echo ============================================================
pause
exit /b 1

:found
echo Found compatible Python %FOUND% - creating virtual environment in venv...
py -%FOUND% -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist assets mkdir assets
if not exist exports mkdir exports
if not exist assets\hand_landmarker.task (
    echo Downloading hand landmark model into assets...
    curl -L -o assets\hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

echo.
echo ============================================================
echo  Setup complete! AirDraw AI is ready on Windows.
echo.
echo  To run AirDraw AI:
echo      venv\Scripts\activate
echo      python airdraw.py
echo.
echo  To run test suite:
echo      pytest -v
echo ============================================================
pause
exit /b 0
