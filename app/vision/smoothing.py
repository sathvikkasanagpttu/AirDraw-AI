"""
AirDraw AI - Coordinate Smoothing & Stroke Interpolation
========================================================
Implements One-Euro velocity-adaptive filtering and sub-pixel stroke interpolation
to eliminate camera sensor jitter and prevent broken lines during rapid strokes.
"""

import math
import time
from typing import List, Optional, Tuple

Point2D = Tuple[int, int]
PointFloat = Tuple[float, float]


class LowPassFilter:
    """Standard 1st-order low-pass exponential filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last_val: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.last_val is None:
            self.last_val = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self.last_val
        self.last_val = filtered
        return filtered

    def reset(self) -> None:
        self.last_val = None


class OneEuroFilter1D:
    """
    1-D One-Euro Filter.
    Adapts cutoff frequency dynamically based on signal rate of change (speed).
    Slow speeds -> high filtering (anti-jitter).
    Fast speeds -> low filtering (zero lag).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        now = timestamp if timestamp is not None else time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            return self.x_filter.filter(x, 1.0)

        dt = max(now - self.last_time, 1e-5)
        self.last_time = now

        # Compute derivative (rate of change)
        dx = (x - (self.x_filter.last_val or x)) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class AdaptivePointSmoother:
    """
    Dual-axis (X, Y) velocity-adaptive point smoother with jitter deadzone.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.008,
        deadzone_px: float = 1.5,
        fallback_ema_alpha: float = 0.55,
    ):
        self.filter_x = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.deadzone_px = deadzone_px
        self.fallback_ema_alpha = fallback_ema_alpha
        self.last_smoothed_point: Optional[Point2D] = None

    def smooth(self, raw_point: Point2D, timestamp: Optional[float] = None) -> Point2D:
        """Filter a raw pixel coordinate."""
        rx, ry = raw_point

        if self.last_smoothed_point is None:
            sx = int(round(self.filter_x.filter(float(rx), timestamp)))
            sy = int(round(self.filter_y.filter(float(ry), timestamp)))
            self.last_smoothed_point = (sx, sy)
            return self.last_smoothed_point

        # Deadzone check to silence micro-tremor
        dist_sq = (rx - self.last_smoothed_point[0]) ** 2 + (ry - self.last_smoothed_point[1]) ** 2
        if dist_sq < (self.deadzone_px ** 2):
            return self.last_smoothed_point

        sx = int(round(self.filter_x.filter(float(rx), timestamp)))
        sy = int(round(self.filter_y.filter(float(ry), timestamp)))
        self.last_smoothed_point = (sx, sy)
        return self.last_smoothed_point

    def reset(self) -> None:
        """Reset internal filter states when stroke ends or tracking drops."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.last_smoothed_point = None


def interpolate_stroke(p1: Point2D, p2: Point2D, step_px: float = 5.0) -> List[Point2D]:
    """
    Interpolates linear intermediate pixel coordinates between two points
    spaced evenly by step_px to ensure smooth, gap-free stroke rendering.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    if dist <= step_px or dist == 0:
        return [p2]

    num_steps = int(dist // step_px)
    interpolated: List[Point2D] = []
    for i in range(1, num_steps + 1):
        t = i / (num_steps + 1)
        ix = int(round(x1 + t * dx))
        iy = int(round(y1 + t * dy))
        interpolated.append((ix, iy))
    interpolated.append(p2)
    return interpolated
