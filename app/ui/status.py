"""
AirDraw AI - Live System Status HUD
===================================
Professional telemetry card rendering camera state, hand tracking fidelity,
recognized gestures, real-time FPS, and processing latency.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np

from app.vision.gesture_engine import GestureType, HandState


class StatusPanelRenderer:
    """Renders a sleek, high-contrast system telemetry HUD card."""

    @staticmethod
    def draw_status_hud(
        frame: np.ndarray,
        x: int,
        y: int,
        camera_online: bool,
        is_demo: bool,
        hand_state: HandState,
        gesture: GestureType,
        fps: float,
        latency_ms: float,
        active_tool_label: str,
        active_color_bgr: Tuple[int, int, int],
        card_w: int = 420,
        card_h: int = 100,
    ) -> None:
        """Draws telemetry card with glassmorphic backdrop and color status dots."""
        # Semi-transparent dark container card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Subtle card border
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (50, 55, 68), 1)

        # Title bar
        cv2.putText(
            frame,
            "SYSTEM TELEMETRY",
            (x + 14, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )

        # Row 1: Camera & Tracking
        # Camera indicator
        if is_demo:
            cam_txt, cam_dot = "Demo Simulation", (0, 200, 255)  # Cyan
        elif camera_online:
            cam_txt, cam_dot = "Connected", (60, 220, 80)        # Green
        else:
            cam_txt, cam_dot = "Disconnected", (50, 50, 230)     # Red

        cv2.circle(frame, (x + 18, y + 42), 4, cam_dot, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Camera: {cam_txt}",
            (x + 28, y + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # Hand Tracking indicator
        if hand_state == HandState.HAND_DETECTED:
            track_txt, track_dot = "Active", (60, 220, 80)
        elif hand_state == HandState.TRACKING_LOST:
            track_txt, track_dot = "Lost", (0, 140, 255)         # Orange
        elif hand_state == HandState.UNCERTAIN:
            track_txt, track_dot = "Uncertain", (0, 210, 255)    # Yellow
        else:
            track_txt, track_dot = "Searching...", (140, 140, 140)

        cv2.circle(frame, (x + 200, y + 42), 4, track_dot, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Tracking: {track_txt}",
            (x + 210, y + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # Row 2: Gesture & Metrics
        # Gesture mapping with emoji labels
        gesture_labels = {
            GestureType.DRAW: "DRAW (Index)",
            GestureType.PEN_UP: "PEN UP (Two Fingers)",
            GestureType.PAUSE: "PAUSE (Open Palm)",
            GestureType.CONFIRM: "CONFIRM (Thumbs Up)",
            GestureType.CLEAR: "CLEAR (Fist)",
            GestureType.PINCH: "PINCH (Select)",
            GestureType.HOVER: "HOVER (Point)",
            GestureType.NONE: "IDLE",
        }
        g_txt = gesture_labels.get(gesture, "NONE")
        cv2.putText(
            frame,
            f"Gesture: {g_txt}",
            (x + 14, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Performance Stats (FPS & Latency)
        perf_txt = f"{fps:.0f} FPS  |  {latency_ms:.0f} ms"
        cv2.putText(
            frame,
            perf_txt,
            (x + card_w - 120, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 175, 190),
            1,
            cv2.LINE_AA,
        )

        # Row 3: Active Tool chip
        cv2.circle(frame, (x + 20, y + 89), 5, active_color_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + 20, y + 89), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Tool: {active_tool_label}",
            (x + 32, y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 195, 210),
            1,
            cv2.LINE_AA,
        )
