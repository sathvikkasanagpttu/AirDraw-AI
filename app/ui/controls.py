"""
AirDraw AI - UI Controls & Hover-to-Click Buttons
=================================================
Interactive, touchless control widgets with visual hover feedback,
radial/linear progress fill, and mouse event routing.
"""

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ButtonWidget:
    """Touchless UI button definition."""
    id: str
    label: str
    text: str
    x1: int
    y1: int
    x2: int
    y2: int
    bg_color: Tuple[int, int, int]
    kind: str = "action"  # "color", "tool", "brush_type", "action"
    is_active: bool = False
    is_repeating: bool = False
    tooltip: str = ""

    def contains(self, px: int, py: int) -> bool:
        """Determines if given pixel coordinate falls inside button boundary."""
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


class HoverInteractionManager:
    """Manages touchless fingertip hovering, dwell-time progress, and auto-repeat."""

    def __init__(self, hover_duration_sec: float = 0.45, repeat_interval_sec: float = 0.22):
        self.hover_duration_sec = hover_duration_sec
        self.repeat_interval_sec = repeat_interval_sec
        self.hovered_button_id: Optional[str] = None
        self.hover_start_time: Optional[float] = None
        self.has_fired: bool = False
        self.last_fire_time: float = 0.0

    def update(self, tip_coords: Optional[Tuple[int, int]], buttons: List[ButtonWidget]) -> Optional[str]:
        """
        Updates dwell timer based on fingertip position.
        Returns the button ID that fired an action, if any.
        """
        now = time.time()
        if tip_coords is None:
            self._reset()
            return None

        px, py = tip_coords
        hit_button: Optional[ButtonWidget] = None
        for b in buttons:
            if b.contains(px, py):
                hit_button = b
                break

        if hit_button is None:
            self._reset()
            return None

        # Switched to a new button
        if hit_button.id != self.hovered_button_id:
            self.hovered_button_id = hit_button.id
            self.hover_start_time = now
            self.has_fired = False
            return None

        # Dwell time reached
        dwell_time = now - (self.hover_start_time or now)
        if not self.has_fired:
            if dwell_time >= self.hover_duration_sec:
                self.has_fired = True
                self.last_fire_time = now
                return hit_button.id
        elif hit_button.is_repeating and (now - self.last_fire_time >= self.repeat_interval_sec):
            self.last_fire_time = now
            return hit_button.id

        return None

    def get_progress(self) -> float:
        """Returns 0.0 to 1.0 dwell completion ratio for active hovered button."""
        if self.hover_start_time is None or self.hovered_button_id is None:
            return 0.0
        if self.has_fired:
            return 1.0
        elapsed = time.time() - self.hover_start_time
        return float(np.clip(elapsed / self.hover_duration_sec, 0.0, 1.0))

    def _reset(self) -> None:
        self.hovered_button_id = None
        self.hover_start_time = None
        self.has_fired = False


def render_button(
    frame: np.ndarray,
    button: ButtonWidget,
    is_hovered: bool = False,
    hover_progress: float = 0.0,
) -> None:
    """Renders a sleek, rounded dashboard-style button with hover animation."""
    x1, y1, x2, y2 = button.x1, button.y1, button.x2, button.y2
    w = x2 - x1
    h = y2 - y1

    # Base background card
    fill_col = button.bg_color
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_col, -1)

    # Active highlight border
    if button.is_active:
        border_col = (255, 255, 255)
        thick = 3
    elif is_hovered:
        border_col = (0, 240, 255)
        thick = 2
    else:
        border_col = (45, 45, 50)
        thick = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, thick)

    # Hover progress fill bar at bottom of card
    if is_hovered and hover_progress > 0:
        bar_w = int(w * hover_progress)
        cv2.rectangle(frame, (x1, y2 - 4), (x1 + bar_w, y2), (255, 255, 255), -1)

    # Text label centered
    dark_text = (
        fill_col[0] * 0.114 + fill_col[1] * 0.587 + fill_col[2] * 0.299 > 160
    )
    text_col = (15, 15, 15) if dark_text else (245, 245, 245)

    font_scale = 0.48
    (tw, th), _ = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    tx = x1 + (w - tw) // 2
    ty = y1 + (h + th) // 2
    cv2.putText(frame, button.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 1, cv2.LINE_AA)
