#!/usr/bin/env bash
# =============================================================================
# AirDraw AI: AI-Powered Air Drawing & Gesture Interaction Platform
# Automated Environment Setup for macOS and Linux
# =============================================================================
set -e

echo "============================================================"
echo " Setting up AirDraw AI Environment..."
echo "============================================================"

FOUND=""
for v in 3.12 3.11 3.10 3.9; do
    if command -v "python$v" >/dev/null 2>&1; then
        FOUND="python$v"
        break
    fi
done

if [ -z "$FOUND" ]; then
    echo "============================================================"
    echo " Error: Compatible Python interpreter not found (need 3.9 - 3.12)."
    echo " MediaPipe currently targets Python 3.9-3.12 runtimes."
    echo ""
    echo " Install Python 3.12:"
    echo "   macOS (Homebrew):  brew install python@3.12"
    echo "   Ubuntu / Debian:   sudo apt install python3.12 python3.12-venv"
    echo "============================================================"
    exit 1
fi

echo "Found compatible Python interpreter: $FOUND"
echo "Creating isolated virtual environment in ./venv..."
"$FOUND" -m venv --clear venv
source venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

echo "Upgrading pip and installing production dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Ensure assets directory and model exist
mkdir -p assets exports
if [ ! -f "assets/hand_landmarker.task" ]; then
    echo "Downloading MediaPipe HandLandmarker model to assets/..."
    curl -L -o assets/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo ""
echo "============================================================"
echo " Setup complete! AirDraw AI is ready."
echo ""
echo " To launch the platform:"
echo "     source venv/bin/activate"
echo "     python airdraw.py"
echo ""
echo " To run the test suite:"
echo "     pytest -v"
echo "============================================================"
