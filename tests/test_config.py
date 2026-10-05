"""Unit tests for configuration validation and settings containers."""

import pytest
from app.config.settings import (
    AppConfig,
    load_default_config,
    CameraConfig,
    VisionConfig,
    SmoothingConfig,
    GestureConfig,
    DrawingConfig,
)


def test_default_config_loading():
    config = load_default_config()
    assert isinstance(config, AppConfig)
    assert config.camera.width > 0
    assert config.camera.height > 0
    assert config.camera.fps > 0

    assert 0.0 < config.vision.min_detection_confidence <= 1.0
    assert 0.0 < config.vision.min_presence_confidence <= 1.0
    assert 0.0 < config.vision.min_tracking_confidence <= 1.0

    assert 0.0 < config.gestures.pinch_threshold_norm < 1.0
    assert config.gestures.fist_hold_frames >= 1

    assert config.drawing.min_brush_size < config.drawing.max_brush_size
    assert config.drawing.export_dir.exists()


def test_color_palette_structure():
    config = load_default_config()
    palette = config.COLOR_PALETTE
    assert len(palette) >= 6

    for name, bgr in palette.items():
        assert isinstance(name, str) and len(name) > 0
        assert len(bgr) == 3
        assert all(0 <= channel <= 255 for channel in bgr)
