import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSnapshot:
    display_fps: float
    prediction_fps: float
    display_p95_ms: float
    frame_count: int
    prediction_count: int

    @property
    def has_activity(self):
        return bool(self.frame_count or self.prediction_count)

    def format_log(self):
        return (
            "[Performance] "
            f"display={self.display_fps:.1f} fps, "
            f"prediction={self.prediction_fps:.1f} fps, "
            f"UI display p95={self.display_p95_ms:.2f} ms"
        )


class PerformanceMetrics:
    """Maintain a bounded sliding window of UI-side throughput metrics."""

    def __init__(self, window_s=5.0, clock=None, max_samples=10000):
        self.window_s = max(0.1, float(window_s))
        self._clock = clock or time.perf_counter
        self._started_at = self._clock()
        self._frame_times = deque(maxlen=max_samples)
        self._prediction_times = deque(maxlen=max_samples)
        self._display_samples = deque(maxlen=max_samples)

    def record_frame(self, display_seconds, now=None):
        now = self._clock() if now is None else float(now)
        self._frame_times.append(now)
        self._display_samples.append((now, max(0.0, float(display_seconds)) * 1000.0))

    def record_prediction(self, now=None):
        now = self._clock() if now is None else float(now)
        self._prediction_times.append(now)

    def snapshot(self, now=None):
        now = self._clock() if now is None else float(now)
        cutoff = now - self.window_s
        _discard_older_than(self._frame_times, cutoff)
        _discard_older_than(self._prediction_times, cutoff)
        while self._display_samples and self._display_samples[0][0] < cutoff:
            self._display_samples.popleft()
        elapsed = max(0.001, min(self.window_s, now - self._started_at))
        durations = [sample[1] for sample in self._display_samples]
        return PerformanceSnapshot(
            display_fps=len(self._frame_times) / elapsed,
            prediction_fps=len(self._prediction_times) / elapsed,
            display_p95_ms=_percentile(durations, 95.0),
            frame_count=len(self._frame_times),
            prediction_count=len(self._prediction_times),
        )


def _discard_older_than(values, cutoff):
    while values and values[0] < cutoff:
        values.popleft()


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return float(ordered[index])
