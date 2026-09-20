import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSnapshot:
    display_fps: float
    prediction_fps: float
    display_p95_ms: float
    payload_build_p95_ms: float
    queue_wait_p95_ms: float
    zmq_p95_ms: float
    inference_p95_ms: float
    end_to_end_p95_ms: float
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
            f"UI p95={self.display_p95_ms:.2f} ms, "
            f"payload p95={self.payload_build_p95_ms:.2f} ms, "
            f"queue p95={self.queue_wait_p95_ms:.2f} ms, "
            f"ZMQ p95={self.zmq_p95_ms:.2f} ms, "
            f"ONNX p95={self.inference_p95_ms:.2f} ms, "
            f"end-to-end p95={self.end_to_end_p95_ms:.2f} ms"
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
        self._latency_samples = {
            name: deque(maxlen=max_samples)
            for name in (
                "payload_build",
                "queue_wait",
                "zmq",
                "inference",
                "end_to_end",
            )
        }

    def record_frame(self, display_seconds, now=None):
        now = self._clock() if now is None else float(now)
        self._frame_times.append(now)
        self._display_samples.append((now, max(0.0, float(display_seconds)) * 1000.0))

    def record_prediction(self, latency_ms=None, now=None):
        now = self._clock() if now is None else float(now)
        self._prediction_times.append(now)
        if isinstance(latency_ms, dict):
            for name, samples in self._latency_samples.items():
                value = latency_ms.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    samples.append((now, float(value)))

    def snapshot(self, now=None):
        now = self._clock() if now is None else float(now)
        cutoff = now - self.window_s
        _discard_older_than(self._frame_times, cutoff)
        _discard_older_than(self._prediction_times, cutoff)
        while self._display_samples and self._display_samples[0][0] < cutoff:
            self._display_samples.popleft()
        for samples in self._latency_samples.values():
            while samples and samples[0][0] < cutoff:
                samples.popleft()
        elapsed = max(0.001, min(self.window_s, now - self._started_at))
        durations = [sample[1] for sample in self._display_samples]
        latency_p95 = {
            name: _percentile([sample[1] for sample in samples], 95.0)
            for name, samples in self._latency_samples.items()
        }
        return PerformanceSnapshot(
            display_fps=len(self._frame_times) / elapsed,
            prediction_fps=len(self._prediction_times) / elapsed,
            display_p95_ms=_percentile(durations, 95.0),
            payload_build_p95_ms=latency_p95["payload_build"],
            queue_wait_p95_ms=latency_p95["queue_wait"],
            zmq_p95_ms=latency_p95["zmq"],
            inference_p95_ms=latency_p95["inference"],
            end_to_end_p95_ms=latency_p95["end_to_end"],
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
