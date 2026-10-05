"""
AirDraw AI - Brush Styles & Stroke Renderers
============================================
Implements diverse drawing styles: Classic, Neon Glow, Highlighter, and Calligraphy.
"""

from enum import Enum
import math
from typing import List, Tuple

import cv2
import numpy as np

Point2D = Tuple[int, int]
ColorBGR = Tuple[int, int, int]


class BrushType(Enum):
    """Available drawing brush aesthetics."""
    CLASSIC = "Classic"
    NEON = "Neon"
    HIGHLIGHTER = "Highlighter"
    CALLIGRAPHY = "Calligraphy"
    ERASER = "Eraser"


class BrushRenderer:
    """Renders artistic strokes across coordinate sequences on a canvas layer."""

    @staticmethod
    def render_point(
        layer: np.ndarray,
        point: Point2D,
        color: ColorBGR,
        size: int,
        brush_type: BrushType = BrushType.CLASSIC,
    ) -> None:
        """Renders an individual dab or point on the canvas."""
        radius = max(1, size // 2)

        if brush_type == BrushType.CLASSIC:
            cv2.circle(layer, point, radius, color, -1, cv2.LINE_AA)

        elif brush_type == BrushType.NEON:
            # Outer bloom halo
            halo_radius = int(radius * 1.8)
            overlay = layer.copy()
            cv2.circle(overlay, point, halo_radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.45, layer, 0.55, 0, layer)
            # Bright inner core
            core_color = (
                min(255, int(color[0] * 0.4 + 180)),
                min(255, int(color[1] * 0.4 + 180)),
                min(255, int(color[2] * 0.4 + 180)),
            )
            cv2.circle(layer, point, radius, core_color, -1, cv2.LINE_AA)

        elif brush_type == BrushType.HIGHLIGHTER:
            overlay = layer.copy()
            cv2.circle(overlay, point, int(radius * 1.4), color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, layer, 0.65, 0, layer)

        elif brush_type == BrushType.CALLIGRAPHY:
            # Elliptical rotated dab
            cv2.ellipse(layer, point, (radius * 2, max(2, radius // 2)), 45, 0, 360, color, -1, cv2.LINE_AA)

        elif brush_type == BrushType.ERASER:
            cv2.circle(layer, point, radius, (0, 0, 0), -1, cv2.LINE_AA)

    @staticmethod
    def render_segment(
        layer: np.ndarray,
        p1: Point2D,
        p2: Point2D,
        color: ColorBGR,
        size: int,
        brush_type: BrushType = BrushType.CLASSIC,
    ) -> None:
        """Renders an interpolated line segment between two consecutive points."""
        thickness = max(1, size)

        if brush_type == BrushType.CLASSIC:
            cv2.line(layer, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.circle(layer, p2, thickness // 2, color, -1, cv2.LINE_AA)

        elif brush_type == BrushType.NEON:
            # Outer glowing aura
            halo_thick = int(thickness * 2.2)
            overlay = layer.copy()
            cv2.line(overlay, p1, p2, color, halo_thick, cv2.LINE_AA)
            cv2.circle(overlay, p2, halo_thick // 2, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.4, layer, 0.6, 0, layer)

            # Vibrant inner core
            core_color = (
                min(255, int(color[0] * 0.3 + 190)),
                min(255, int(color[1] * 0.3 + 190)),
                min(255, int(color[2] * 0.3 + 190)),
            )
            cv2.line(layer, p1, p2, core_color, max(1, int(thickness * 0.6)), cv2.LINE_AA)
            cv2.circle(layer, p2, max(1, int(thickness * 0.3)), core_color, -1, cv2.LINE_AA)

        elif brush_type == BrushType.HIGHLIGHTER:
            overlay = layer.copy()
            cv2.line(overlay, p1, p2, color, int(thickness * 1.6), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, layer, 0.65, 0, layer)

        elif brush_type == BrushType.CALLIGRAPHY:
            # Render angled ribbon segment
            angle_rad = math.radians(45)
            dx = int((thickness // 2) * math.cos(angle_rad))
            dy = int((thickness // 2) * math.sin(angle_rad))
            pts = np.array([
                [p1[0] - dx, p1[1] - dy],
                [p1[0] + dx, p1[1] + dy],
                [p2[0] + dx, p2[1] + dy],
                [p2[0] - dx, p2[1] - dy],
            ], dtype=np.int32)
            cv2.fillConvexPoly(layer, pts, color, cv2.LINE_AA)

        elif brush_type == BrushType.ERASER:
            cv2.line(layer, p1, p2, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.circle(layer, p2, thickness // 2, (0, 0, 0), -1, cv2.LINE_AA)
