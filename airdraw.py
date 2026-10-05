"""
AirDraw AI - Main Entry Point
=============================
AI-Powered Air Drawing & Gesture Interaction Platform.

For modular architecture and subsystems, see the `app/` package.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.makedirs("/tmp/mpl_cache", exist_ok=True)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import AirDrawApplication, main

# Backwards compatibility aliases
AirDraw = AirDrawApplication

if __name__ == "__main__":
    main()
