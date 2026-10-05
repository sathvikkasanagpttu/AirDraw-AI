"""Integration tests for AirDrawApplication end-to-end lifecycle."""

from pathlib import Path
import numpy as np
import pytest
from app.config.settings import AppConfig, load_default_config
from app.drawing.brush import BrushType
from app.main import AirDrawApplication
from app.vision.gesture_engine import GestureType


def test_app_lifecycle_demo_mode(tmp_path: Path):
    # Configure app with temporary export path and demo mode
    config = load_default_config()
    config.drawing.export_dir = tmp_path
    config.camera.demo_mode = True

    app = AirDrawApplication(config=config)
    assert app.is_demo_mode is True
    assert app.camera_online is False

    # Simulate single frame cycle
    frame, tip = app._generate_demo_frame()
    assert frame.shape == (config.camera.height, config.camera.width, 3)
    assert tip is not None

    # Test stroke creation in demo mode
    app.canvas.add_point(tip)
    assert app.canvas.is_stroking is True
    app.canvas.end_stroke()
    assert app.canvas.is_stroking is False

    # Test action dispatchers
    app._execute_action("brush_type")
    assert app.canvas.brush_type in list(BrushType)

    app._execute_action("color_Electric Blue")
    assert app.canvas.active_color_name == "Electric Blue"

    app._execute_action("eraser")
    assert app.canvas.is_eraser is True

    app._execute_action("eraser")
    assert app.canvas.is_eraser is False

    # Test undo / redo dispatch
    app._execute_action("undo")
    app._execute_action("redo")

    # Test save action
    app._execute_action("save")
    saved_files = list(tmp_path.glob("airdraw_*.png"))
    assert len(saved_files) >= 1

    # Test key handlers
    assert app._handle_key(ord('c')) is True
    assert app._handle_key(ord('h')) is True
    assert app._handle_key(ord('q')) is False  # Q signals exit

    # Safe shutdown
    app.close()
    assert app.is_running is False
