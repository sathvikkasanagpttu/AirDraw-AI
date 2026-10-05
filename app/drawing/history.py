"""
AirDraw AI - Drawing History & Undo/Redo Engine
===============================================
Manages state snapshots with memory-bounded undo/redo stacks.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np


class HistoryManager:
    """Manages undo and redo stacks for multiple drawing layers."""

    def __init__(self, max_history: int = 25):
        self.max_history = max_history
        self._undo_stack: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_history)
        self._redo_stack: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_history)

    def initialize(self, canvas: np.ndarray, calc_canvas: np.ndarray) -> None:
        """Sets the baseline canvas state."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_stack.append((canvas.copy(), calc_canvas.copy()))

    def push(self, canvas: np.ndarray, calc_canvas: np.ndarray) -> None:
        """Pushes a new state after stroke completion or layer alteration."""
        self._undo_stack.append((canvas.copy(), calc_canvas.copy()))
        self._redo_stack.clear()

    def undo(self, current_canvas: np.ndarray, current_calc: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Pops previous state and preserves current state on redo stack."""
        if len(self._undo_stack) <= 1:
            return None

        # Pop current state to redo stack
        curr_state = self._undo_stack.pop()
        self._redo_stack.append(curr_state)

        # Restore top of undo stack
        target = self._undo_stack[-1]
        return target[0].copy(), target[1].copy()

    def redo(self, current_canvas: np.ndarray, current_calc: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Re-applies the most recently reverted state."""
        if not self._redo_stack:
            return None

        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        return state[0].copy(), state[1].copy()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
