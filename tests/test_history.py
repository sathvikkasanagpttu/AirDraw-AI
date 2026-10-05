"""Unit tests for canvas history and undo/redo operations."""

import numpy as np
import pytest
from app.drawing.history import HistoryManager


def test_history_initialization():
    hist = HistoryManager(max_history=5)
    c1 = np.zeros((10, 10, 3), dtype=np.uint8)
    calc1 = np.zeros((10, 10, 3), dtype=np.uint8)

    hist.initialize(c1, calc1)
    assert not hist.can_undo
    assert not hist.can_redo


def test_history_push_and_undo_redo():
    hist = HistoryManager(max_history=5)
    c0 = np.zeros((10, 10, 3), dtype=np.uint8)
    calc0 = np.zeros((10, 10, 3), dtype=np.uint8)
    hist.initialize(c0, calc0)

    # State 1
    c1 = c0.copy()
    c1[0, 0] = [255, 0, 0]
    hist.push(c1, calc0)

    assert hist.can_undo
    assert not hist.can_redo

    # Undo to State 0
    restored = hist.undo(c1, calc0)
    assert restored is not None
    restored_c, _ = restored
    assert restored_c[0, 0, 0] == 0
    assert not hist.can_undo
    assert hist.can_redo

    # Redo back to State 1
    redone = hist.redo(restored_c, calc0)
    assert redone is not None
    redone_c, _ = redone
    assert redone_c[0, 0, 0] == 255
    assert hist.can_undo
    assert not hist.can_redo


def test_history_max_limit():
    max_h = 3
    hist = HistoryManager(max_history=max_h)
    c = np.zeros((10, 10, 3), dtype=np.uint8)
    calc = np.zeros((10, 10, 3), dtype=np.uint8)
    hist.initialize(c, calc)

    for i in range(10):
        c_new = c.copy()
        c_new[0, 0] = i
        hist.push(c_new, calc)

    # Stack size should be capped at max_h
    assert len(hist._undo_stack) <= max_h


def test_history_empty_safety():
    hist = HistoryManager(max_history=5)
    c = np.zeros((5, 5, 3), dtype=np.uint8)
    assert hist.undo(c, c) is None
    assert hist.redo(c, c) is None
