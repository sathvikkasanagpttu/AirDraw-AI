"""
AirDraw AI - Hand Tracking Subsystem
====================================
Integrates MediaPipe Tasks HandLandmarker with robust state tracking,
graceful error handling, and high-tech landmark rendering.
"""

from dataclasses import dataclass
import os
import time
import urllib.request
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from app.config.settings import VisionConfig
from app.utils.logger import get_logger
from app.vision.gesture_engine import HandState

logger = get_logger("HandTracker")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


@dataclass
class TrackingResult:
    """Consolidated hand detection output."""
    state: HandState
    landmarks: Optional[List[Any]] = None
    pixel_landmarks: Optional[List[Tuple[int, int]]] = None
    index_tip: Optional[Tuple[int, int]] = None
    thumb_tip: Optional[Tuple[int, int]] = None
    handedness: str = "Right"
    confidence: float = 0.0
    inference_time_sec: float = 0.0


class HandTracker:
    """High-level wrapper around MediaPipe HandLandmarker."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.landmarker: Optional[mp_vision.HandLandmarker] = None
        self._frame_timestamp: int = 0
        self._last_detection_time: float = 0.0
        self._is_initialized: bool = False

        self._ensure_model_file()
        self._initialize_detector()

    def _ensure_model_file(self) -> None:
        """Download model bundle if not already present on local disk."""
        model_path = self.config.model_path
        if os.path.exists(model_path):
            return

        logger.info(f"Hand landmark model not found locally. Downloading from {self.config.model_url}...")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        try:
            urllib.request.urlretrieve(self.config.model_url, model_path)
            logger.info(f"Model downloaded successfully to {model_path}")
        except Exception as exc:
            logger.error(f"Failed to download hand landmark model: {exc}")
            raise RuntimeError(f"Could not download hand tracking model: {exc}") from exc

    def _initialize_detector(self) -> None:
        """Initialize the MediaPipe Landmarker task graph."""
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=self.config.model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.config.max_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe HandLandmarker task graph initialized.")
        except Exception as exc:
            self._is_initialized = False
            logger.error(f"MediaPipe initialization error: {exc}")
            raise RuntimeError(f"Hand tracker initialization failed: {exc}") from exc

    def process_frame(self, bgr_frame: np.ndarray) -> TrackingResult:
        """
        Executes vision inference on an input frame and calculates tracking states.
        """
        if not self._is_initialized or self.landmarker is None:
            return TrackingResult(state=HandState.HAND_NOT_DETECTED)

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_timestamp += 1
        t_start = time.perf_counter()
        detection_result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp)
        inference_duration = time.perf_counter() - t_start

        now = time.time()

        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            landmarks = detection_result.hand_landmarks[0]
            self._last_detection_time = now

            # Handedness label
            handedness = "Right"
            if detection_result.handedness and detection_result.handedness[0]:
                handedness = detection_result.handedness[0][0].category_name

            # Pixel conversions
            pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            idx_tip = pixel_pts[8]
            th_tip = pixel_pts[4]

            # Confidence determination
            conf = 0.90
            if detection_result.handedness and detection_result.handedness[0]:
                conf = float(detection_result.handedness[0][0].score)

            state = HandState.HAND_DETECTED if conf >= self.config.uncertainty_threshold else HandState.UNCERTAIN

            return TrackingResult(
                state=state,
                landmarks=landmarks,
                pixel_landmarks=pixel_pts,
                index_tip=idx_tip,
                thumb_tip=th_tip,
                handedness=handedness,
                confidence=conf,
                inference_time_sec=inference_duration,
            )

        # Hand not detected in current frame: distinguish between LOST and INACTIVE
        if now - self._last_detection_time < 1.0 and self._last_detection_time > 0:
            state = HandState.TRACKING_LOST
        else:
            state = HandState.HAND_NOT_DETECTED

        return TrackingResult(
            state=state,
            inference_time_sec=inference_duration,
        )

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pixel_landmarks: List[Tuple[int, int]],
        active_color: Tuple[int, int, int] = (0, 220, 255),
    ) -> None:
        """
        Renders a modern, subtle cybernetic hand skeleton.
        """
        # Bones
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixel_landmarks[a], pixel_landmarks[b], (45, 180, 80), 2, cv2.LINE_AA)

        # Joints
        for i, pt in enumerate(pixel_landmarks):
            radius = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, pt, radius, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

        # Index tip glowing aura
        idx_tip = pixel_landmarks[8]
        cv2.circle(frame, idx_tip, 12, active_color, 2, cv2.LINE_AA)
        cv2.circle(frame, idx_tip, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
            self._is_initialized = False
            logger.info("Hand tracker resources successfully released.")
