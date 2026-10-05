"""Unit tests for adaptive smoothing and stroke interpolation."""

import pytest
from app.vision.smoothing import (
    LowPassFilter,
    OneEuroFilter1D,
    AdaptivePointSmoother,
    interpolate_stroke,
)


def test_low_pass_filter():
    lpf = LowPassFilter(alpha=0.5)
    assert lpf.filter(10.0) == 10.0  # Initial value untouched
    assert lpf.filter(20.0) == 15.0  # 0.5 * 20 + 0.5 * 10 = 15.0
    lpf.reset()
    assert lpf.last_val is None
    assert lpf.filter(50.0) == 50.0


def test_one_euro_filter():
    one_euro = OneEuroFilter1D(min_cutoff=1.0, beta=0.01)
    t = 0.0
    v1 = one_euro.filter(100.0, timestamp=t)
    assert v1 == 100.0

    # Small tremor: should be heavily smoothed
    t += 0.033
    v2 = one_euro.filter(101.0, timestamp=t)
    assert 100.0 < v2 < 101.0

    # Rapid motion: should track fast
    t += 0.033
    v3 = one_euro.filter(300.0, timestamp=t)
    assert v3 > 150.0


def test_adaptive_point_smoother():
    smoother = AdaptivePointSmoother(deadzone_px=2.0)
    p1 = smoother.smooth((100, 200))
    assert p1 == (100, 200)

    # Sub-pixel / deadzone jitter (< 2px distance) should snap to previous point
    p2 = smoother.smooth((101, 200))
    assert p2 == (100, 200)

    # Significant movement (> 2px distance) should update
    p3 = smoother.smooth((150, 250))
    assert p3[0] > 100 and p3[1] > 200

    smoother.reset()
    assert smoother.last_smoothed_point is None


def test_interpolate_stroke():
    # Coincident or sub-step points
    pts_close = interpolate_stroke((10, 10), (10, 10), step_px=5.0)
    assert pts_close == [(10, 10)]

    # Interpolation along line segment of length 20px
    p1 = (0, 0)
    p2 = (20, 0)
    interpolated = interpolate_stroke(p1, p2, step_px=5.0)
    assert len(interpolated) >= 4
    assert interpolated[-1] == (20, 0)
    # Check monotonic progression along X axis
    xs = [p[0] for p in interpolated]
    assert xs == sorted(xs)
