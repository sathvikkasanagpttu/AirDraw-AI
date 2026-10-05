"""
AirDraw AI - Performance Monitoring
===================================
Lightweight, real-time FPS and latency tracking with rolling averages.
"""

import time
from collections import deque
from typing import Dict, Any


class PerformanceMonitor:
    """Tracks frame processing latency and frames-per-second."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._frame_times = deque(maxlen=window_size)
        self._inference_latencies = deque(maxlen=window_size)
        self._last_tick = time.perf_counter()
        self._fps: float = 0.0
        self._avg_latency_ms: float = 0.0
        self._avg_inference_ms: float = 0.0

    def start_frame(self) -> None:
        """Mark start of frame pipeline."""
        self._last_tick = time.perf_counter()

    def record_inference(self, duration_sec: float) -> None:
        """Record hand detection / model inference time."""
        self._inference_latencies.append(duration_sec * 1000.0)
        if self._inference_latencies:
            self._avg_inference_ms = sum(self._inference_latencies) / len(self._inference_latencies)

    def end_frame(self) -> None:
        """Mark end of frame and update rolling metrics."""
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        if dt > 0:
            self._frame_times.append(dt)
            total_time = sum(self._frame_times)
            if total_time > 0:
                self._fps = len(self._frame_times) / total_time
            self._avg_latency_ms = (total_time / len(self._frame_times)) * 1000.0

    @property
    def fps(self) -> float:
        """Current smoothed FPS."""
        return self._fps

    @property
    def latency_ms(self) -> float:
        """Average total frame latency in ms."""
        return self._avg_latency_ms

    @property
    def inference_latency_ms(self) -> float:
        """Average model inference latency in ms."""
        return self._avg_inference_ms

    def get_metrics(self) -> Dict[str, Any]:
        """Summary dictionary for UI status bar."""
        return {
            "fps": round(self._fps, 1),
            "latency_ms": round(self._avg_latency_ms, 1),
            "inference_ms": round(self._avg_inference_ms, 1),
        }
