"""Unit tests for performance monitoring and latency calculation."""

import time
import pytest
from app.utils.performance import PerformanceMonitor


def test_performance_monitor_fps_and_latency():
    monitor = PerformanceMonitor(window_size=10)
    assert monitor.fps == 0.0

    # Simulate 5 frames spaced by 10ms
    for _ in range(5):
        monitor.start_frame()
        monitor.record_inference(0.005)  # 5ms
        time.sleep(0.005)
        monitor.end_frame()

    assert monitor.fps > 0.0
    assert monitor.latency_ms > 0.0
    assert monitor.inference_latency_ms > 0.0

    metrics = monitor.get_metrics()
    assert "fps" in metrics
    assert "latency_ms" in metrics
    assert "inference_ms" in metrics
