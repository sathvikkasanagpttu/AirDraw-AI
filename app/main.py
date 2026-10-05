"""
AirDraw AI - Main Application Controller
========================================
Production-style orchestrator connecting camera ingestion, MediaPipe hand tracking,
adaptive coordinate smoothing, gesture recognition, canvas rendering, and UI.
"""

import math
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.settings import AppConfig, load_default_config
from app.drawing.brush import BrushType
from app.drawing.canvas import CanvasEngine
from app.math_recognition.recognizer import MathRecognizer
from app.ui.interface import UIManager
from app.utils.logger import get_logger
from app.utils.performance import PerformanceMonitor
from app.vision.gesture_engine import GestureEngine, GestureType, HandState
from app.vision.hand_tracker import HandTracker, TrackingResult
from app.vision.smoothing import AdaptivePointSmoother

logger = get_logger("AirDrawAI")


class AirDrawApplication:
    """Master application controller for AirDraw AI."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_default_config()

        # Performance & telemetry
        self.perf = PerformanceMonitor()
        self.is_running: bool = False
        self.is_demo_mode: bool = False
        self.camera_online: bool = False

        # Vision & Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_w: int = self.config.camera.width
        self.frame_h: int = self.config.camera.height
        self._demo_tick: float = 0.0

        self._setup_camera()

        # Core subsystems
        self.tracker: Optional[HandTracker] = None
        if not self.is_demo_mode:
            try:
                self.tracker = HandTracker(self.config.vision)
            except Exception as exc:
                logger.warning(f"Hand tracker unavailable ({exc}). Engaging Demo Simulation Mode.")
                self.is_demo_mode = True
        self.gesture_engine = GestureEngine(
            pinch_threshold=self.config.gestures.pinch_threshold_norm,
            hold_frames=self.config.gestures.fist_hold_frames,
            cooldown_sec=self.config.gestures.toggle_cooldown_sec,
        )
        self.smoother = AdaptivePointSmoother(
            min_cutoff=self.config.smoothing.min_cutoff,
            beta=self.config.smoothing.beta,
        )
        self.canvas = CanvasEngine(self.frame_w, self.frame_h, self.config.drawing)
        self.recognizer = MathRecognizer()
        self.ui = UIManager(self.frame_w, self.frame_h, self.config)

        # Operational state
        self.is_draw_enabled: bool = True
        self.calc_auto: bool = True
        self.calc_result = None
        self.smoothed_tip: Optional[Tuple[int, int]] = None

        logger.info("AirDraw AI platform fully initialized.")

    def _setup_camera(self) -> None:
        """Initializes webcam capture or gracefully defaults to demo simulation."""
        cam_idx = self.config.camera.index
        if self.config.camera.demo_mode or cam_idx < 0:
            logger.info("Demo mode engaged via configuration. Camera hardware bypassed.")
            self.camera_online = False
            self.is_demo_mode = True
            return

        try:
            self.cap = cv2.VideoCapture(cam_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {cam_idx}.")

            ok, test_frame = self.cap.read()
            if not ok or test_frame is None:
                raise RuntimeError("Failed to read test frame from camera.")

            self.frame_h, self.frame_w = test_frame.shape[:2]
            self.camera_online = True
            self.is_demo_mode = False
            logger.info(f"Camera successfully connected at {self.frame_w}x{self.frame_h}")

        except Exception as exc:
            logger.warning(f"Camera unavailable ({exc}). Engaging Demo Simulation Mode.")
            self.camera_online = False
            self.is_demo_mode = True
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def _generate_demo_frame(self) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Synthesizes high-contrast studio background with moving simulated cursor."""
        frame = np.full((self.frame_h, self.frame_w, 3), (24, 26, 32), dtype=np.uint8)
        self._demo_tick += 0.04

        # Animated Lissajous trajectory
        cx = self.frame_w // 2
        cy = self.frame_h // 2 + 30
        sim_x = int(cx + 260 * math.sin(self._demo_tick * 0.7))
        sim_y = int(cy + 160 * math.sin(self._demo_tick * 1.4))
        tip = (sim_x, sim_y)

        # Draw demo pointer
        cv2.circle(frame, tip, 12, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, tip, 4, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "[DEMO SIMULATION - CAMERA OFFLINE]",
            (self.frame_w // 2 - 190, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
        return frame, tip

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Optional[object]) -> None:
        """Processes direct mouse clicks on UI sidebar controls."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.ui.show_help:
                self.ui.show_help = False
                return

            hit_btn = self.ui.hit_test(x, y)
            if hit_btn is not None:
                self._execute_action(hit_btn.id)

    def _execute_action(self, action_id: str) -> None:
        """Executes interactive actions triggered by touchless hover, mouse, or keyboard."""
        if action_id.startswith("color_"):
            color_name = action_id.replace("color_", "")
            bgr = self.config.COLOR_PALETTE.get(color_name, (240, 220, 0))
            self.canvas.set_color(color_name, bgr)
            self.ui.toast.show(f"Color: {color_name}", color=(40, 160, 60))

        elif action_id == "eraser":
            is_on = self.canvas.toggle_eraser()
            state = "Enabled" if is_on else "Disabled"
            self.ui.toast.show(f"Eraser {state}", color=(120, 120, 120))

        elif action_id == "calc":
            self.canvas.is_calc_mode = not self.canvas.is_calc_mode
            if self.canvas.is_calc_mode:
                self.canvas.is_eraser = False
                self.canvas.calc_dirty = False
                self.calc_result = None
                self.ui.toast.show("Calculator Mode Activated", color=(0, 180, 220))
            else:
                self.ui.toast.show("Drawing Mode Restored", color=(0, 180, 80))

        elif action_id == "brush_type":
            # Cycle through brush types
            types = list(BrushType)
            cur_idx = types.index(self.canvas.brush_type)
            next_type = types[(cur_idx + 1) % len(types)]
            if next_type == BrushType.ERASER:
                next_type = BrushType.CLASSIC
            self.canvas.set_brush_type(next_type)
            self.ui.toast.show(f"Brush: {next_type.value}", color=(0, 200, 240))

        elif action_id == "undo":
            if self.canvas.undo():
                self.ui.toast.show("Action Reverted (Undo)", color=(70, 70, 90))

        elif action_id == "redo":
            if self.canvas.redo():
                self.ui.toast.show("Action Restored (Redo)", color=(70, 70, 90))

        elif action_id == "clear":
            self.canvas.clear()
            self.calc_result = None
            self.ui.toast.show("Canvas Cleared", color=(180, 50, 50))

        elif action_id == "save":
            saved_path = self.canvas.export_png(transparent=False)
            self.ui.toast.show(f"Saved: {saved_path.name}", color=(40, 160, 60), duration_sec=3.0)

        elif action_id == "export_alpha":
            saved_path = self.canvas.export_png(transparent=True)
            self.ui.toast.show(f"Saved Alpha: {saved_path.name}", color=(140, 50, 140), duration_sec=3.0)

        elif action_id == "size_minus":
            new_sz = self.canvas.change_brush_size(-2)
            self.ui.toast.show(f"Brush Size: {new_sz}px", color=(60, 80, 100), duration_sec=1.0)

        elif action_id == "size_plus":
            new_sz = self.canvas.change_brush_size(2)
            self.ui.toast.show(f"Brush Size: {new_sz}px", color=(60, 80, 100), duration_sec=1.0)

    def _handle_key(self, key: int) -> bool:
        """Processes keyboard hotkeys. Returns False if application should exit."""
        if key in (ord('q'), 27):  # Q or ESC
            return False

        if key in (ord(' '), ord('h')):
            self.ui.show_help = not self.ui.show_help
            return True

        if ord('1') <= key <= ord('8'):
            colors = list(self.config.COLOR_PALETTE.keys())
            idx = key - ord('1')
            if idx < len(colors):
                self._execute_action(f"color_{colors[idx]}")

        elif key == ord('b'):
            self._execute_action("brush_type")
        elif key == ord('e'):
            self._execute_action("eraser")
        elif key == ord('k'):
            self._execute_action("calc")
        elif key == ord('c'):
            self._execute_action("clear")
        elif key == ord('u'):
            self._execute_action("undo")
        elif key == ord('r'):
            self._execute_action("redo")
        elif key == ord('s'):
            self._execute_action("save")
        elif key == ord('t'):
            self._execute_action("export_alpha")
        elif key in (ord('+'), ord('=')):
            self._execute_action("size_plus")
        elif key in (ord('-'), ord('_')):
            self._execute_action("size_minus")
        elif key == ord('m'):
            self.is_draw_enabled = not self.is_draw_enabled
            state = "ON" if self.is_draw_enabled else "PAUSED (Menu)"
            self.ui.toast.show(f"Draw Mode: {state}", color=(0, 180, 220))

        return True

    def _update_calculator(self) -> None:
        """Asynchronously parses and solves handwritten math equations on canvas."""
        if not (self.canvas.is_calc_mode and self.canvas.calc_dirty):
            return

        now = time.time()
        # Evaluate after 0.8s pause in handwriting
        if now - self.canvas.last_ink_timestamp < 0.8:
            return

        ink_mask = (np.max(self.canvas.calc_layer, axis=2) > 0).astype(np.uint8) * 255
        self.calc_result = self.recognizer.read_ink(ink_mask)
        self.canvas.calc_dirty = False

    def run(self) -> None:
        """Main application video ingestion and rendering loop."""
        win_name = self.config.ui.window_name
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self.on_mouse)

        self.is_running = True
        logger.info("AirDraw AI interaction loop started.")

        while self.is_running:
            self.perf.start_frame()

            # 1. Acquire frame
            if self.camera_online and self.cap is not None:
                ok, raw_frame = self.cap.read()
                if not ok or raw_frame is None:
                    logger.warning("Frame read failure. Falling back to simulation.")
                    frame, sim_tip = self._generate_demo_frame()
                else:
                    if self.config.camera.mirror:
                        frame = cv2.flip(raw_frame, 1)
                    else:
                        frame = raw_frame
                    sim_tip = None
            else:
                frame, sim_tip = self._generate_demo_frame()

            # 2. Hand tracking & vision inference
            if not self.is_demo_mode:
                track_res = self.tracker.process_frame(frame)
                self.perf.record_inference(track_res.inference_time_sec)
                raw_tip = track_res.index_tip
                hand_state = track_res.state
                landmarks = track_res.landmarks
                handedness = track_res.handedness
                tracking_conf = track_res.confidence
            else:
                track_res = TrackingResult(state=HandState.HAND_DETECTED)
                raw_tip = sim_tip
                hand_state = HandState.HAND_DETECTED
                landmarks = None
                handedness = "Right"
                tracking_conf = 0.95

            # 3. Gesture recognition
            if landmarks is not None:
                gesture, conf = self.gesture_engine.classify(
                    landmarks, handedness=handedness, tracking_confidence=tracking_conf
                )
            elif self.is_demo_mode:
                gesture = GestureType.DRAW
            else:
                gesture = GestureType.NONE

            # 4. Handle gestures
            # Open palm toggles Draw / Pause
            if gesture == GestureType.PAUSE:
                if self.gesture_engine.can_trigger_action(GestureType.PAUSE):
                    self.is_draw_enabled = False
                    self.canvas.end_stroke()

            # Closed fist triggers clear with cooldown
            elif gesture == GestureType.CLEAR:
                if self.gesture_engine.can_trigger_action(GestureType.CLEAR):
                    self._execute_action("clear")

            # Thumbs up triggers math solve
            elif gesture == GestureType.CONFIRM:
                if self.gesture_engine.can_trigger_action(GestureType.CONFIRM):
                    if self.canvas.is_calc_mode:
                        self.canvas.calc_dirty = True
                        self.canvas.last_ink_timestamp = 0.0

            # 5. Coordinate smoothing & stroke generation
            if raw_tip is not None and self.ui.in_draw_area(raw_tip):
                self.smoothed_tip = self.smoother.smooth(raw_tip)
            else:
                self.smoother.reset()
                self.smoothed_tip = None

            # Drawing execution
            should_draw = (
                self.is_draw_enabled
                and (gesture == GestureType.DRAW or (self.is_demo_mode and sim_tip is not None))
                and not self.ui.show_help
            )

            if should_draw and self.smoothed_tip is not None and self.ui.in_draw_area(self.smoothed_tip):
                self.canvas.add_point(self.smoothed_tip)
            else:
                self.canvas.end_stroke()

            # Touchless hover selection when in PAUSED or HOVER mode
            if not self.is_draw_enabled and raw_tip is not None:
                fired_action = self.ui.update_touchless_hover(raw_tip)
                if fired_action is not None:
                    self._execute_action(fired_action)
            else:
                self.ui.update_touchless_hover(None)

            # 6. Background math calculator update
            self._update_calculator()

            # 7. Rendering pipeline: Camera Frame -> Hand Skeleton -> Canvas Layers -> UI HUD
            if (
                self.config.vision.show_landmarks
                and self.tracker is not None
                and track_res.pixel_landmarks
            ):
                self.tracker.draw_skeleton(
                    frame,
                    track_res.pixel_landmarks,
                    active_color=self.canvas.active_color,
                )

            # Composite drawing ink
            self.canvas.composite_onto_frame(frame)

            # Eraser fingertip guide ring
            if self.canvas.is_eraser and self.smoothed_tip and self.ui.in_draw_area(self.smoothed_tip):
                er_size = self.canvas.effective_brush_size
                cv2.circle(frame, self.smoothed_tip, er_size // 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render UI controls, telemetry, and overlays
            self.ui.render(
                frame=frame,
                active_color_name=self.canvas.active_color_name,
                is_eraser=self.canvas.is_eraser,
                is_calc=self.canvas.is_calc_mode,
                brush_type=self.canvas.brush_type,
                brush_size=self.canvas.brush_size,
                is_draw_enabled=self.is_draw_enabled,
                hand_state=hand_state,
                gesture=gesture,
                fps=self.perf.fps,
                latency_ms=self.perf.latency_ms,
                camera_online=self.camera_online,
                is_demo=self.is_demo_mode,
                calc_result=self.calc_result,
                calc_dirty=self.canvas.calc_dirty,
                calc_auto=self.calc_auto,
                tip_coords=raw_tip,
            )

            cv2.imshow(win_name, frame)
            self.perf.end_frame()

            # Keyboard event handling
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if not self._handle_key(key):
                    break

        self.close()

    def close(self) -> None:
        """Orderly shutdown and resource cleanup."""
        logger.info("Terminating AirDraw AI application...")
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        if self.tracker is not None:
            self.tracker.close()
        logger.info("All resources safely released.")


def main() -> None:
    """Standard console entrypoint with CLI flag parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AirDraw AI - AI-Powered Air Drawing & Gesture Interaction Platform"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch in synthetic demo simulation mode without webcam hardware",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target camera frame width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target camera frame height (default: 720)",
    )
    args = parser.parse_args()

    cfg = load_default_config()
    cfg.camera.demo_mode = args.demo
    cfg.camera.index = args.cam
    cfg.camera.width = args.width
    cfg.camera.height = args.height

    app = AirDrawApplication(config=cfg)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    finally:
        app.close()


if __name__ == "__main__":
    main()
