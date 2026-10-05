"""Unit tests for virtual canvas engine, brush dynamics, and image exports."""

from pathlib import Path
import numpy as np
import pytest
import cv2
from app.drawing.brush import BrushType
from app.drawing.canvas import CanvasEngine
from app.config.settings import DrawingConfig


def test_canvas_initialization():
    engine = CanvasEngine(width=640, height=480)
    assert engine.main_layer.shape == (480, 640, 3)
    assert engine.calc_layer.shape == (480, 640, 3)
    assert not engine.is_stroking


def test_canvas_stroke_rendering():
    engine = CanvasEngine(width=200, height=200)
    engine.set_color("TestRed", (0, 0, 255))
    engine.brush_size = 6

    # Draw a line from (50, 50) to (100, 50)
    engine.add_point((50, 50))
    assert engine.is_stroking
    engine.add_point((100, 50))
    engine.end_stroke()
    assert not engine.is_stroking

    # Check ink exists on canvas
    drawn_pixels = np.count_nonzero(engine.main_layer)
    assert drawn_pixels > 0
    # Center pixel should have red channel (BGR idx 2)
    assert engine.main_layer[50, 75, 2] > 200


def test_brush_size_bounds():
    cfg = DrawingConfig(min_brush_size=4, max_brush_size=20, default_brush_size=10)
    engine = CanvasEngine(width=100, height=100, config=cfg)

    # Increment within bounds
    engine.change_brush_size(5)
    assert engine.brush_size == 15

    # Increment past upper bound
    engine.change_brush_size(50)
    assert engine.brush_size == 20

    # Decrement past lower bound
    engine.change_brush_size(-100)
    assert engine.brush_size == 4


def test_canvas_clear():
    engine = CanvasEngine(width=100, height=100)
    engine.add_point((20, 20))
    engine.add_point((30, 30))
    engine.end_stroke()
    assert np.count_nonzero(engine.main_layer) > 0

    engine.clear()
    assert np.count_nonzero(engine.main_layer) == 0


def test_canvas_export(tmp_path: Path):
    cfg = DrawingConfig(export_dir=tmp_path)
    engine = CanvasEngine(width=100, height=100, config=cfg)
    engine.add_point((20, 20))
    engine.add_point((40, 40))
    engine.end_stroke()

    # Standard 3-channel PNG export
    path_std = engine.export_png(transparent=False)
    assert path_std.exists()
    img_std = cv2.imread(str(path_std))
    assert img_std.shape == (100, 100, 3)

    # Transparent 4-channel BGRA export
    path_alpha = engine.export_png(transparent=True)
    assert path_alpha.exists()
    img_alpha = cv2.imread(str(path_alpha), cv2.IMREAD_UNCHANGED)
    assert img_alpha.shape == (100, 100, 4)
    # Background pixels should have alpha = 0
    assert img_alpha[0, 0, 3] == 0
    # Drawn pixel should have alpha = 255
    assert img_alpha[30, 30, 3] == 255
