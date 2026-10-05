"""Unit tests for gesture recognition and debouncing state machine."""

from dataclasses import dataclass
from typing import List
import pytest
from app.vision.gesture_engine import GestureEngine, GestureType


@dataclass
class MockLandmark:
    x: float
    y: float
    z: float = 0.0


def create_mock_hand(
    thumb_ext: bool = False,
    index_ext: bool = False,
    middle_ext: bool = False,
    ring_ext: bool = False,
    pinky_ext: bool = False,
    pinch: bool = False,
    thumbs_up: bool = False,
) -> List[MockLandmark]:
    """Generates synthetic 21 hand landmarks matching requested finger states."""
    # Base wrist at bottom center
    lms = [MockLandmark(0.5, 0.8) for _ in range(21)]

    # Thumb: 1, 2, 3, 4
    lms[2] = MockLandmark(0.42, 0.70)
    lms[3] = MockLandmark(0.38, 0.65)
    if pinch:
        # Put thumb tip right at index tip
        lms[4] = MockLandmark(0.50, 0.40)
    elif thumbs_up:
        # Thumb pointing up high
        lms[4] = MockLandmark(0.35, 0.40)
    elif thumb_ext:
        lms[4] = MockLandmark(0.30, 0.60)
    else:
        lms[4] = MockLandmark(0.40, 0.68)

    # Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
    lms[5] = MockLandmark(0.46, 0.60)
    lms[6] = MockLandmark(0.46, 0.52)
    lms[7] = MockLandmark(0.46, 0.46)
    if index_ext or pinch:
        lms[8] = MockLandmark(0.46, 0.38)
    else:
        lms[8] = MockLandmark(0.46, 0.64)

    # Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
    lms[9] = MockLandmark(0.50, 0.58)
    lms[10] = MockLandmark(0.50, 0.50)
    lms[11] = MockLandmark(0.50, 0.44)
    if middle_ext:
        lms[12] = MockLandmark(0.50, 0.36)
    else:
        lms[12] = MockLandmark(0.50, 0.62)

    # Ring: 13(MCP), 14(PIP), 15(DIP), 16(TIP)
    lms[13] = MockLandmark(0.54, 0.60)
    lms[14] = MockLandmark(0.54, 0.52)
    lms[15] = MockLandmark(0.54, 0.46)
    if ring_ext:
        lms[16] = MockLandmark(0.54, 0.38)
    else:
        lms[16] = MockLandmark(0.54, 0.64)

    # Pinky: 17(MCP), 18(PIP), 19(DIP), 20(TIP)
    lms[17] = MockLandmark(0.58, 0.62)
    lms[18] = MockLandmark(0.58, 0.55)
    lms[19] = MockLandmark(0.58, 0.50)
    if pinky_ext:
        lms[20] = MockLandmark(0.58, 0.42)
    else:
        lms[20] = MockLandmark(0.58, 0.66)

    return lms


def test_gesture_draw_classification():
    engine = GestureEngine(hold_frames=1)
    hand = create_mock_hand(index_ext=True)
    gesture, conf = engine.classify(hand)
    assert gesture == GestureType.DRAW
    assert conf > 0.8


def test_gesture_pause_open_palm():
    engine = GestureEngine(hold_frames=1)
    hand = create_mock_hand(
        thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
    )
    gesture, conf = engine.classify(hand)
    assert gesture == GestureType.PAUSE
    assert conf > 0.8


def test_gesture_pen_up_two_fingers():
    engine = GestureEngine(hold_frames=1)
    hand = create_mock_hand(index_ext=True, middle_ext=True)
    gesture, conf = engine.classify(hand)
    assert gesture == GestureType.PEN_UP
    assert conf > 0.8


def test_gesture_clear_fist():
    engine = GestureEngine(hold_frames=1)
    hand = create_mock_hand()  # All curled
    gesture, conf = engine.classify(hand)
    assert gesture == GestureType.CLEAR
    assert conf > 0.8


def test_gesture_pinch():
    engine = GestureEngine(hold_frames=1, pinch_threshold=0.08)
    hand = create_mock_hand(pinch=True)
    gesture, conf = engine.classify(hand)
    assert gesture == GestureType.PINCH


def test_gesture_cooldown():
    engine = GestureEngine(hold_frames=1, cooldown_sec=0.5)
    hand = create_mock_hand()  # Fist / Clear
    gesture, _ = engine.classify(hand)
    assert gesture == GestureType.CLEAR

    # First trigger should succeed
    assert engine.can_trigger_action(GestureType.CLEAR) is True

    # Immediate second trigger should be blocked by cooldown
    assert engine.can_trigger_action(GestureType.CLEAR) is False


def test_missing_landmarks():
    engine = GestureEngine(hold_frames=1)
    gesture, conf = engine.classify(None)
    assert gesture == GestureType.NONE
    assert conf == 0.0
