"""
AirDraw AI - Modal Overlays, Onboarding, Toasts & Calculator Banner
===================================================================
Renders onboarding tutorials, quick-reference gesture guides,
animated toast notifications, and mathematical formula overlay cards.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.math_recognition.recognizer import MathRecognitionResult


class ToastNotifier:
    """Manages temporary feedback flash notifications."""

    def __init__(self):
        self.message: Optional[str] = None
        self.color: Tuple[int, int, int] = (40, 160, 60)
        self.expires_at: float = 0.0

    def show(self, text: str, duration_sec: float = 2.5, color: Tuple[int, int, int] = (40, 160, 60)) -> None:
        self.message = text
        self.color = color
        self.expires_at = time.time() + duration_sec

    def draw(self, frame: np.ndarray) -> None:
        if not self.message or time.time() >= self.expires_at:
            return

        h, w = frame.shape[:2]
        font_scale = 0.65
        thick = 2
        (tw, th), _ = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

        card_w = tw + 40
        card_h = th + 24
        cx = w // 2
        cy = 130
        x1 = cx - card_w // 2
        y1 = cy - card_h // 2
        x2 = cx + card_w // 2
        y2 = cy + card_h // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(frame, self.message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)


class OnboardingOverlay:
    """Full-screen onboarding modal tutorial with gesture cheatsheet."""

    @staticmethod
    def draw(frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        # Darken entire frame for modal focus
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (12, 14, 18), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Main modal window container
        mw, mh = min(920, w - 80), min(560, h - 60)
        mx1 = (w - mw) // 2
        my1 = (h - mh) // 2
        mx2 = mx1 + mw
        my2 = my1 + mh

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (26, 30, 38), -1)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 210, 255), 2)

        # Modal Header
        cv2.putText(
            frame,
            "AirDraw AI",
            (mx1 + 36, my1 + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "AI-Powered Air Drawing & Gesture Interaction Platform",
            (mx1 + 38, my1 + 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 190, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.line(frame, (mx1 + 36, my1 + 90), (mx2 - 36, my1 + 90), (50, 55, 70), 1)

        # Gestures Guide (Left Column)
        gx = mx1 + 40
        gy = my1 + 120
        cv2.putText(frame, "HAND GESTURE RECOGNITION", (gx, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        gestures = [
            ("1. DRAW", "Raise 1 Index Finger to paint fluid lines in the air"),
            ("2. PAUSE / MENU", "Open Palm (5 fingers) to pause and access UI sidebar"),
            ("3. PEN UP", "Two Fingers (Index + Middle) to hover without painting"),
            ("4. CONFIRM", "Thumbs Up to solve air arithmetic or confirm actions"),
            ("5. CLEAR", "Closed Fist held for ~0.5s to clear canvas"),
            ("6. PINCH", "Touch Thumb & Index tips together for precision pick"),
        ]

        row_y = gy + 32
        for title, desc in gestures:
            cv2.putText(frame, title, (gx, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (gx + 150, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 205, 220), 1, cv2.LINE_AA)
            row_y += 32

        # Keyboard Cheatsheet (Right Column / Bottom)
        kx = mx1 + 40
        ky = row_y + 16
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        shortcuts_col1 = [
            "[1 - 8]  Select Color",
            "[B]      Cycle Brush Type (Neon, Classic, ...)",
            "[E]      Toggle Eraser",
            "[K]      Toggle Math Calculator",
        ]
        shortcuts_col2 = [
            "[C]      Clear Canvas",
            "[U / R]  Undo / Redo",
            "[S / T]  Save PNG / Transparent PNG",
            "[+ / -]  Brush Size",
        ]

        sy = ky + 30
        for s1, s2 in zip(shortcuts_col1, shortcuts_col2):
            cv2.putText(frame, s1, (kx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 250), 1, cv2.LINE_AA)
            cv2.putText(frame, s2, (kx + 380, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 250), 1, cv2.LINE_AA)
            sy += 26

        # Dismiss footer
        dismiss_txt = "Press [SPACE] or [H] to begin drawing"
        (dtw, dth), _ = cv2.getTextSize(dismiss_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        btn_x = (mx1 + mx2 - dtw) // 2
        btn_y = my2 - 24
        cv2.putText(frame, dismiss_txt, (btn_x, btn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 160), 1, cv2.LINE_AA)


class MathBannerRenderer:
    """Renders top floating card when Math Calculator mode is active."""

    @staticmethod
    def draw(
        frame: np.ndarray,
        result: Optional[MathRecognitionResult],
        is_dirty: bool,
        auto_calc: bool,
        center_x: int,
    ) -> None:
        pw, ph = 540, 68
        x1 = center_x - pw // 2
        y1 = 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + pw, y1 + ph), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), (0, 210, 255), 2)

        if result is None:
            msg = "MATH CALC: write arithmetic (e.g. 12 + 5)"
            col = (210, 210, 210)
            if is_dirty:
                msg = "MATH CALC: evaluating..." if auto_calc else "MATH CALC: Thumbs Up or tap Solve"
            cv2.putText(frame, msg, (x1 + 20, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2, cv2.LINE_AA)
            return

        expr = result.shown
        for op in "+-*/":
            expr = expr.replace(op, f" {op} ")
        expr = expr.replace("*", "x").replace("  ", " ").strip()

        if result.value is not None:
            msg = f"{expr} = {result.value}"
            col = (90, 255, 120)
        else:
            msg = f"{expr or result.text}  (syntax error)"
            col = (60, 180, 255)

        cv2.putText(frame, msg, (x1 + 20, y1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.88, col, 2, cv2.LINE_AA)

        # Symbol tags above recognized glyphs
        for sym, (bx1, by1, bx2, by2) in zip(result.symbols, result.boxes):
            label = "x" if sym == "*" else sym
            cx = bx1 + (bx2 - bx1) // 2 - 6
            cy = max(110, by1 - 10)
            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 255), 2, cv2.LINE_AA)

        # Big answer adjacent to writing
        if result.value is not None:
            bx1, by1, bx2, by2 = result.bbox
            scale = float(np.clip((by2 - by1) / 45.0, 1.0, 2.5))
            thick = max(2, int(2 * scale))
            ans_text = result.value if "=" in result.text else f"= {result.value}"
            (tw, th), _ = cv2.getTextSize(ans_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            tx = bx2 + 30
            ty = (by1 + by2) // 2 + th // 2
            # Outlined text
            cv2.putText(frame, ans_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
            cv2.putText(frame, ans_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (90, 255, 120), thick, cv2.LINE_AA)
