"""
AirDraw AI - Configuration Module
=================================
Centralized, validated configuration settings for AirDraw AI platform.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
EXPORTS_DIR = PROJECT_ROOT / "exports"


@dataclass
class CameraConfig:
    """Camera acquisition settings."""
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    auto_fallback_demo: bool = True
    demo_mode: bool = False


@dataclass
class VisionConfig:
    """Hand tracking and landmark detection settings."""
    model_path: str = field(default_factory=lambda: str(
        ASSETS_DIR / "hand_landmarker.task"
        if (ASSETS_DIR / "hand_landmarker.task").exists()
        else PROJECT_ROOT / "hand_landmarker.task"
    ))
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    max_hands: int = 1
    min_detection_confidence: float = 0.65
    min_presence_confidence: float = 0.60
    min_tracking_confidence: float = 0.55
    show_landmarks: bool = True
    uncertainty_threshold: float = 0.50


@dataclass
class SmoothingConfig:
    """Anti-jitter and stroke smoothing parameters."""
    enabled: bool = True
    ema_alpha: float = 0.55
    adaptive: bool = True
    min_cutoff: float = 1.0  # Hz for One-Euro style filtering
    beta: float = 0.007      # Speed coefficient
    d_cutoff: float = 1.0
    interpolation_step_px: float = 6.0


@dataclass
class GestureConfig:
    """Gesture recognition rules and debouncing thresholds."""
    enabled: bool = True
    pinch_threshold_norm: float = 0.065
    fist_hold_frames: int = 4
    open_palm_hold_frames: int = 4
    toggle_cooldown_sec: float = 0.7
    hover_click_time_sec: float = 0.45
    repeat_interval_sec: float = 0.22


@dataclass
class DrawingConfig:
    """Canvas, brush styles, and history configuration."""
    min_brush_size: int = 2
    max_brush_size: int = 50
    default_brush_size: int = 8
    eraser_size_multiplier: int = 4
    min_calc_pen_size: int = 3
    max_calc_pen_size: int = 20
    default_calc_pen_size: int = 8
    max_history_states: int = 25
    export_dir: Path = field(default_factory=lambda: EXPORTS_DIR)
    default_color_name: str = "Cyan"
    default_brush_type: str = "Classic"  # "Classic", "Neon", "Highlighter", "Calligraphy"


@dataclass
class UIConfig:
    """Visual theme and user interface metrics."""
    window_name: str = "AirDraw AI - Air Drawing & Gesture Interaction Platform"
    panel_width: int = 135
    bottom_bar_height: int = 44
    button_margin: int = 10
    button_gap: int = 8
    max_button_height: int = 68
    show_onboarding: bool = True
    dark_mode_bg: Tuple[int, int, int] = (18, 18, 22)
    accent_color: Tuple[int, int, int] = (0, 210, 255)  # Cyan BGR


@dataclass
class AppConfig:
    """Master application configuration container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Standard Color Palette (Name -> BGR Tuple)
    COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "Electric Blue": (255, 120, 0),     # Vibrant Blue
        "Neon Green": (80, 225, 60),        # Vibrant Green
        "Coral Red": (60, 60, 235),         # Vivid Red
        "Cyber Yellow": (0, 220, 245),      # Vivid Yellow
        "Purple Glow": (215, 50, 195),      # Neon Violet
        "Cyan": (240, 220, 0),              # Electric Cyan
        "Studio White": (250, 250, 250),    # Pure Off-White
        "Deep Amber": (20, 140, 255),       # Orange/Amber
    })

    # Available Brush Types
    BRUSH_TYPES: List[str] = field(default_factory=lambda: [
        "Classic",
        "Neon",
        "Highlighter",
        "Calligraphy",
    ])


def load_default_config() -> AppConfig:
    """Create and return standard verified config instance."""
    config = AppConfig()
    config.drawing.export_dir.mkdir(parents=True, exist_ok=True)
    return config
