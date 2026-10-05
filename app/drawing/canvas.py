"""
AirDraw AI - Virtual Drawing Canvas Engine
==========================================
Coordinates multi-layer rendering, artistic stroke generation,
stroke interpolation, state history, and transparent PNG exports.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.settings import DrawingConfig, EXPORTS_DIR
from app.drawing.brush import BrushRenderer, BrushType, ColorBGR, Point2D
from app.drawing.history import HistoryManager
from app.utils.logger import get_logger
from app.vision.smoothing import interpolate_stroke

logger = get_logger("CanvasEngine")


class CanvasEngine:
    """Manages virtual drawing layers, stroke lifecycles, and export pipelines."""

    def __init__(self, width: int, height: int, config: Optional[DrawingConfig] = None):
        self.width = width
        self.height = height
        self.config = config or DrawingConfig()

        # Canvas raster buffers (Height x Width x 3 BGR)
        self.main_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.calc_layer = np.zeros((height, width, 3), dtype=np.uint8)

        # Drawing attributes
        self.active_color: ColorBGR = (240, 220, 0)  # Default Cyan
        self.active_color_name: str = self.config.default_color_name
        self.brush_size: int = self.config.default_brush_size
        self.brush_type: BrushType = BrushType.CLASSIC
        self.calc_pen_size: int = self.config.default_calc_pen_size
        self.is_eraser: bool = False
        self.is_calc_mode: bool = False

        # Stroke tracking
        self.prev_point: Optional[Point2D] = None
        self.is_stroking: bool = False
        self.calc_dirty: bool = False
        self.last_ink_timestamp: float = 0.0

        # State history
        self.history = HistoryManager(max_history=self.config.max_history_states)
        self.history.initialize(self.main_layer, self.calc_layer)

    @property
    def active_layer(self) -> np.ndarray:
        """Returns the currently targeted drawing layer."""
        return self.calc_layer if self.is_calc_mode else self.main_layer

    @property
    def effective_brush_size(self) -> int:
        """Computes current stroke thickness with modifiers applied."""
        if self.is_eraser:
            return self.brush_size * self.config.eraser_size_multiplier
        if self.is_calc_mode:
            return self.calc_pen_size
        return self.brush_size

    def set_color(self, name: str, bgr: ColorBGR) -> None:
        """Sets active drawing color and disables eraser."""
        self.active_color_name = name
        self.active_color = bgr
        self.is_eraser = False

    def toggle_eraser(self) -> bool:
        """Toggles eraser tool on/off."""
        self.is_eraser = not self.is_eraser
        return self.is_eraser

    def set_brush_type(self, brush_type: BrushType) -> None:
        """Switches active brush aesthetic."""
        self.brush_type = brush_type
        self.is_eraser = False

    def change_brush_size(self, delta: int) -> int:
        """Increments or decrements active brush thickness within bounds."""
        if self.is_calc_mode and not self.is_eraser:
            self.calc_pen_size = int(
                np.clip(
                    self.calc_pen_size + delta,
                    self.config.min_calc_pen_size,
                    self.config.max_calc_pen_size,
                )
            )
            return self.calc_pen_size
        else:
            self.brush_size = int(
                np.clip(
                    self.brush_size + delta,
                    self.config.min_brush_size,
                    self.config.max_brush_size,
                )
            )
            return self.brush_size

    def add_point(self, pt: Point2D) -> None:
        """Appends a new smoothed point to the ongoing stroke with interpolation."""
        target_layer = self.active_layer
        current_brush = BrushType.ERASER if self.is_eraser else self.brush_type
        current_color = (0, 0, 0) if self.is_eraser else self.active_color
        current_size = self.effective_brush_size

        if self.prev_point is None:
            # First point of new stroke
            BrushRenderer.render_point(
                target_layer, pt, current_color, current_size, current_brush
            )
            self.is_stroking = True
            self.prev_point = pt
        else:
            # Interpolate sub-points for smooth, non-choppy strokes
            interpolated = interpolate_stroke(self.prev_point, pt, step_px=4.0)
            p_prev = self.prev_point
            for p_curr in interpolated:
                BrushRenderer.render_segment(
                    target_layer, p_prev, p_curr, current_color, current_size, current_brush
                )
                p_prev = p_curr
            self.prev_point = pt

        if self.is_calc_mode:
            self.calc_dirty = True
            self.last_ink_timestamp = time.time()

    def end_stroke(self) -> None:
        """Concludes active stroke and pushes canvas state to undo stack."""
        if self.is_stroking:
            self.history.push(self.main_layer, self.calc_layer)
            self.is_stroking = False
        self.prev_point = None

    def undo(self) -> bool:
        """Reverts canvas to previous historical state."""
        self.end_stroke()
        restored = self.history.undo(self.main_layer, self.calc_layer)
        if restored is not None:
            self.main_layer, self.calc_layer = restored
            if self.is_calc_mode:
                self.calc_dirty = True
                self.last_ink_timestamp = time.time()
            return True
        return False

    def redo(self) -> bool:
        """Re-applies reverted state from redo stack."""
        self.end_stroke()
        restored = self.history.redo(self.main_layer, self.calc_layer)
        if restored is not None:
            self.main_layer, self.calc_layer = restored
            if self.is_calc_mode:
                self.calc_dirty = True
                self.last_ink_timestamp = time.time()
            return True
        return False

    def clear(self) -> None:
        """Clears the currently active layer and commits to history."""
        self.end_stroke()
        self.active_layer[:] = 0
        if self.is_calc_mode:
            self.calc_dirty = False
        self.history.push(self.main_layer, self.calc_layer)

    def clear_all(self) -> None:
        """Completely purges all drawing and math layers."""
        self.end_stroke()
        self.main_layer[:] = 0
        self.calc_layer[:] = 0
        self.calc_dirty = False
        self.history.push(self.main_layer, self.calc_layer)

    def export_png(self, transparent: bool = False, output_path: Optional[Path] = None) -> Path:
        """
        Saves current drawing as high-resolution PNG image.
        Supports transparent alpha export for stickers / digital art.
        """
        export_folder = self.config.export_dir or EXPORTS_DIR
        export_folder.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Composite main and calc layers
        composite = self.main_layer.copy()
        calc_mask = np.any(self.calc_layer != 0, axis=2)
        composite[calc_mask] = self.calc_layer[calc_mask]

        if transparent:
            # 4-channel BGRA: alpha = 255 where ink exists, 0 where transparent
            ink_mask = np.any(composite != 0, axis=2).astype(np.uint8) * 255
            b, g, r = cv2.split(composite)
            bgra = cv2.merge([b, g, r, ink_mask])
            save_path = output_path or (export_folder / f"airdraw_{timestamp_str}_transparent.png")
            cv2.imwrite(str(save_path), bgra)
            logger.info(f"Exported transparent canvas to {save_path}")
            return save_path
        else:
            save_path = output_path or (export_folder / f"airdraw_{timestamp_str}.png")
            cv2.imwrite(str(save_path), composite)
            logger.info(f"Exported canvas to {save_path}")
            return save_path

    def composite_onto_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Blends active drawing ink onto camera frame with anti-aliasing.
        """
        # Composite main drawing layer
        main_mask = np.any(self.main_layer != 0, axis=2)
        frame[main_mask] = self.main_layer[main_mask]

        # Composite calculator layer
        calc_mask = np.any(self.calc_layer != 0, axis=2)
        frame[calc_mask] = self.calc_layer[calc_mask]

        return frame
