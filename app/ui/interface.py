"""
AirDraw AI - User Interface Subsystem & Dashboard Manager
=========================================================
Coordinates dashboard layout, touchless sidebars, tool buttons,
status telemetry, toasts, and onboarding modals.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config.settings import AppConfig, UIConfig
from app.drawing.brush import BrushType, ColorBGR
from app.ui.controls import ButtonWidget, HoverInteractionManager, render_button
from app.ui.overlay import MathBannerRenderer, OnboardingOverlay, ToastNotifier
from app.ui.status import StatusPanelRenderer
from app.vision.gesture_engine import GestureType, HandState


class UIManager:
    """Master controller for all interactive visual widgets and telemetry displays."""

    def __init__(self, width: int, height: int, config: AppConfig):
        self.width = width
        self.height = height
        self.config = config
        self.ui_cfg = config.ui

        self.buttons: List[ButtonWidget] = []
        self.hover_manager = HoverInteractionManager(
            hover_duration_sec=config.gestures.hover_click_time_sec,
            repeat_interval_sec=config.gestures.repeat_interval_sec,
        )
        self.toast = ToastNotifier()
        self.show_help: bool = self.ui_cfg.show_onboarding

        self._build_layout()

    def _build_layout(self) -> None:
        """Constructs left palette and right tools columns."""
        self.buttons.clear()
        pw = self.ui_cfg.panel_width
        bm = self.ui_cfg.button_margin
        bg = self.ui_cfg.button_gap
        bh_max = self.ui_cfg.max_button_height
        bottom_limit = self.height - self.ui_cfg.bottom_bar_height - bm

        # 1. Left Panel: Color Palette
        colors = list(self.config.COLOR_PALETTE.items())
        n_col = len(colors)
        avail_h = bottom_limit - bm - bg * (n_col - 1)
        bh_col = min(bh_max, avail_h // n_col)
        y = bm
        for name, bgr in colors:
            self.buttons.append(
                ButtonWidget(
                    id=f"color_{name}",
                    label=name,
                    text=name,
                    x1=bm,
                    y1=y,
                    x2=pw - bm,
                    y2=y + bh_col,
                    bg_color=bgr,
                    kind="color",
                )
            )
            y += bh_col + bg

        # 2. Right Panel: Tools & Operations
        tools = [
            ("brush_type", "Brush", (0, 140, 240), "Brush Style"),
            ("eraser", "Eraser", (100, 100, 100), "Eraser"),
            ("calc", "Calc", (0, 150, 200), "Math Calc"),
            ("undo", "Undo", (75, 75, 85), "Undo"),
            ("redo", "Redo", (75, 75, 85), "Redo"),
            ("clear", "Clear", (50, 50, 160), "Clear"),
            ("save", "Save", (40, 140, 50), "Save PNG"),
            ("export_alpha", "PNG (Alpha)", (140, 60, 140), "Export Transparent"),
            ("size_minus", "Size -", (60, 65, 75), "Smaller Brush"),
            ("size_plus", "Size +", (60, 65, 75), "Larger Brush"),
        ]
        n_tools = len(tools)
        avail_th = bottom_limit - bm - bg * (n_tools - 1)
        bh_tool = min(bh_max, avail_th // n_tools)
        ty = bm
        rx1 = self.width - pw + bm
        rx2 = self.width - bm
        for tid, text, bg_col, label in tools:
            is_rep = tid in ("size_minus", "size_plus")
            self.buttons.append(
                ButtonWidget(
                    id=tid,
                    label=label,
                    text=text,
                    x1=rx1,
                    y1=ty,
                    x2=rx2,
                    y2=ty + bh_tool,
                    bg_color=bg_col,
                    kind="tool",
                    is_repeating=is_rep,
                )
            )
            ty += bh_tool + bg

    @property
    def draw_bounds_x(self) -> Tuple[int, int]:
        """Horizontal extent of drawable region between side panels."""
        return self.ui_cfg.panel_width, self.width - self.ui_cfg.panel_width

    def in_draw_area(self, pt: Tuple[int, int]) -> bool:
        """Verifies if coordinate lies inside active drawing viewport."""
        x, y = pt
        x1, x2 = self.draw_bounds_x
        return x1 <= x <= x2 and y < (self.height - self.ui_cfg.bottom_bar_height)

    def hit_test(self, px: int, py: int) -> Optional[ButtonWidget]:
        """Finds button under coordinates."""
        for b in self.buttons:
            if b.contains(px, py):
                return b
        return None

    def update_touchless_hover(self, tip_coords: Optional[Tuple[int, int]]) -> Optional[str]:
        """Feeds fingertip coordinates into dwell-hover engine."""
        return self.hover_manager.update(tip_coords, self.buttons)

    def render(
        self,
        frame: np.ndarray,
        active_color_name: str,
        is_eraser: bool,
        is_calc: bool,
        brush_type: BrushType,
        brush_size: int,
        is_draw_enabled: bool,
        hand_state: HandState,
        gesture: GestureType,
        fps: float,
        latency_ms: float,
        camera_online: bool,
        is_demo: bool,
        calc_result: Optional[Any] = None,
        calc_dirty: bool = False,
        calc_auto: bool = True,
        tip_coords: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Renders comprehensive UI overlay on top of frame and canvas."""
        pw = self.ui_cfg.panel_width
        bottom_y = self.height - self.ui_cfg.bottom_bar_height

        # Side panel glassmorphism backdrops
        alpha = 0.50 if is_draw_enabled else 0.88
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (pw, bottom_y), (14, 16, 20), -1)
        cv2.rectangle(overlay, (self.width - pw, 0), (self.width, bottom_y), (14, 16, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Drawing viewport boundary line
        border_col = (50, 200, 80) if is_draw_enabled else (0, 160, 255)
        cv2.rectangle(frame, (pw, 0), (self.width - pw, bottom_y), border_col, 2)

        # Render sidebar buttons
        hovered_id = self.hover_manager.hovered_button_id
        hover_prog = self.hover_manager.get_progress()

        for b in self.buttons:
            # Active states
            if b.kind == "color":
                b.is_active = (b.label == active_color_name and not is_eraser)
            elif b.id == "eraser":
                b.is_active = is_eraser
                b.text = "Eraser ON" if is_eraser else "Eraser"
            elif b.id == "calc":
                b.is_active = is_calc
                b.text = "Calc ON" if is_calc else "Calc"
            elif b.id == "brush_type":
                b.text = f"Brush: {brush_type.value}"

            is_hov = (b.id == hovered_id)
            render_button(frame, b, is_hovered=is_hov, hover_progress=hover_prog if is_hov else 0.0)

        # Top System Status HUD Card
        active_col_bgr = (
            (0, 0, 0) if is_eraser else self.config.COLOR_PALETTE.get(active_color_name, (240, 220, 0))
        )
        tool_desc = f"{'Eraser' if is_eraser else brush_type.value} ({brush_size}px)"
        StatusPanelRenderer.draw_status_hud(
            frame,
            x=pw + 16,
            y=12,
            camera_online=camera_online,
            is_demo=is_demo,
            hand_state=hand_state,
            gesture=gesture,
            fps=fps,
            latency_ms=latency_ms,
            active_tool_label=tool_desc,
            active_color_bgr=active_col_bgr,
        )

        # Bottom Shortcut Legend
        b_bar = frame.copy()
        cv2.rectangle(b_bar, (0, bottom_y), (self.width, self.height), (12, 14, 18), -1)
        cv2.addWeighted(b_bar, 0.85, frame, 0.15, 0, frame)

        legend_text = (
            "[1-8] Colors | [B] Brush | [E] Eraser | [K] Calc | [C] Clear | [U] Undo | [R] Redo | "
            "[S] Save | [T] Alpha PNG | [M] Pause/Draw | [H] Help | [Q] Exit"
        )
        cv2.putText(
            frame,
            legend_text,
            (16, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 215, 230),
            1,
            cv2.LINE_AA,
        )

        # Math Calculator Banner (if active)
        if is_calc:
            MathBannerRenderer.draw(
                frame,
                result=calc_result,
                is_dirty=calc_dirty,
                auto_calc=calc_auto,
                center_x=(self.width) // 2 + 100,
            )

        # Toasts
        self.toast.draw(frame)

        # Onboarding overlay (if open)
        if self.show_help:
            OnboardingOverlay.draw(frame)
