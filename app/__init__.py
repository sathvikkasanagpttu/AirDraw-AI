"""
AirDraw AI - AI-Powered Air Drawing & Gesture Interaction Platform
==================================================================
Production-grade computer-vision system for real-time hand tracking and air interaction.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

from app.main import AirDrawApplication, main

__version__ = "2.0.0"
__all__ = ["AirDrawApplication", "main"]
