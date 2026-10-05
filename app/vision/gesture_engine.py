"""
AirDraw AI - Advanced Gesture Recognition Engine
================================================
Robust, orientation-aware hand gesture classifier with confidence scoring,
debouncing, state transition guards, and cooldown management.
"""

from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple


class GestureType(Enum):
    """Recognized user interaction gestures."""
    NONE = "NONE"
    DRAW = "DRAW"                    # ☝️ Index raised, others curled
    PEN_UP = "PEN_UP"                # ✌️ Index + Middle raised (move without drawing / switch tool)
    PAUSE = "PAUSE"                  # ✋ Open Palm (pause drawing / open control menu)
    CONFIRM = "CONFIRM"              # 👍 Thumbs Up (confirm / solve)
    CLEAR = "CLEAR"                  # ✊ Fist held (clear canvas / reset)
    PINCH = "PINCH"                  # 🤏 Thumb and Index pinch (select / adjust)
    HOVER = "HOVER"                  # 👆 Index pointing at UI controls


class HandState(Enum):
    """Discrete system tracking states."""
    HAND_DETECTED = "HAND_DETECTED"
    HAND_NOT_DETECTED = "HAND_NOT_DETECTED"
    TRACKING_LOST = "TRACKING_LOST"
    UNCERTAIN = "UNCERTAIN"


class GestureEngine:
    """
    Classifies 3D normalized landmarks into robust interactive gestures.
    Uses distance-to-wrist and angular extension metrics rather than naive y-axis checks.
    """

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    def __init__(
        self,
        pinch_threshold: float = 0.065,
        hold_frames: int = 4,
        cooldown_sec: float = 0.6,
    ):
        self.pinch_threshold = pinch_threshold
        self.hold_frames = hold_frames
        self.cooldown_sec = cooldown_sec

        self.current_gesture: GestureType = GestureType.NONE
        self.last_fired_gesture: GestureType = GestureType.NONE
        self.candidate_gesture: GestureType = GestureType.NONE
        self.candidate_count: int = 0
        self.last_action_time: float = 0.0
        self.confidence: float = 0.0

    @staticmethod
    def _dist(p1: Any, p2: Any) -> float:
        """Euclidean distance between two landmarks in normalized space."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def analyze_fingers(self, landmarks: List[Any], handedness: str = "Right") -> List[bool]:
        """
        Determines extension state for [Thumb, Index, Middle, Ring, Pinky].
        Combines distance from wrist (0) with joint progression to maintain robustness
        under moderate hand tilts and rotations.
        """
        wrist = landmarks[0]
        extended = [False] * 5

        # 1. Thumb analysis: distance from thumb tip to pinky base (MCP 17) & thumb MCP (2)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        pinky_mcp = landmarks[17]

        dist_tip_pinky = self._dist(thumb_tip, pinky_mcp)
        dist_ip_pinky = self._dist(thumb_ip, pinky_mcp)
        dist_tip_wrist = self._dist(thumb_tip, wrist)
        dist_mcp_wrist = self._dist(thumb_mcp, wrist)

        # Extended if tip is clearly further from palm and wrist than inner joints
        extended[0] = (dist_tip_pinky > dist_ip_pinky * 1.12) and (dist_tip_wrist > dist_mcp_wrist * 1.1)

        # 2. Fingers (Index, Middle, Ring, Pinky)
        for i in range(1, 5):
            tip = landmarks[self.FINGER_TIPS[i]]
            pip = landmarks[self.FINGER_PIPS[i]]
            mcp = landmarks[self.FINGER_MCPS[i]]

            d_tip_wrist = self._dist(tip, wrist)
            d_pip_wrist = self._dist(pip, wrist)
            d_tip_mcp = self._dist(tip, mcp)
            d_pip_mcp = self._dist(pip, mcp)

            # Extended if fingertip is further from wrist than PIP and outer segment is long
            is_extended = (d_tip_wrist > d_pip_wrist * 1.12) and (d_tip_mcp > d_pip_mcp * 1.18)
            extended[i] = is_extended

        return extended

    def classify(
        self,
        landmarks: Optional[List[Any]],
        handedness: str = "Right",
        tracking_confidence: float = 1.0,
    ) -> Tuple[GestureType, float]:
        """
        Processes hand landmarks and returns (GestureType, confidence_score).
        """
        if not landmarks or len(landmarks) < 21:
            self.current_gesture = GestureType.NONE
            self.confidence = 0.0
            return GestureType.NONE, 0.0

        fingers = self.analyze_fingers(landmarks, handedness)
        thumb, index, middle, ring, pinky = fingers

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]

        pinch_dist = self._dist(thumb_tip, index_tip)

        raw_gesture = GestureType.NONE
        raw_conf = 0.75 * tracking_confidence

        # 1. Pinch detection (Thumb + Index close together)
        if pinch_dist < self.pinch_threshold:
            raw_gesture = GestureType.PINCH
            raw_conf = max(0.6, 1.0 - (pinch_dist / self.pinch_threshold))

        # 2. Open Palm (All 5 extended) -> PAUSE
        elif thumb and index and middle and ring and pinky:
            raw_gesture = GestureType.PAUSE
            raw_conf = 0.95

        # 3. Fist (All curled) -> CLEAR
        elif not index and not middle and not ring and not pinky:
            # Check for Thumbs Up first
            if thumb and (thumb_tip.y < wrist.y - 0.12):
                raw_gesture = GestureType.CONFIRM
                raw_conf = 0.90
            else:
                raw_gesture = GestureType.CLEAR
                raw_conf = 0.88

        # 4. Draw (Index extended, others curled)
        elif index and not middle and not ring and not pinky:
            raw_gesture = GestureType.DRAW
            raw_conf = 0.92

        # 5. Pen Up / Two Fingers (Index + Middle extended, others curled)
        elif index and middle and not ring and not pinky:
            raw_gesture = GestureType.PEN_UP
            raw_conf = 0.89

        # Debouncing & Stability tracking
        now = time.time()
        if raw_gesture == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1

        if self.candidate_count >= self.hold_frames:
            self.current_gesture = raw_gesture
            self.confidence = raw_conf
        else:
            self.confidence = 0.5 * raw_conf

        return self.current_gesture, self.confidence

    def can_trigger_action(self, gesture: GestureType) -> bool:
        """Enforces action cooldown for discrete gestures (e.g., Clear, Confirm)."""
        now = time.time()
        if gesture in (GestureType.CLEAR, GestureType.CONFIRM):
            if now - self.last_action_time >= self.cooldown_sec and self.current_gesture == gesture:
                self.last_action_time = now
                self.last_fired_gesture = gesture
                return True
            return False
        return True
