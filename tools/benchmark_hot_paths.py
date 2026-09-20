"""Small, dependency-free performance regression checks for UI Event hot paths."""

import argparse
import json
import queue
import statistics
import time

import numpy as np

from backend.adaptive_inference_queue import AdaptiveInferenceQueueConsumer
from backend.event_pipeline import EventWindowSlicer, InferenceWindow
from backend.event_processing import EVENT_CD_DTYPE, build_inference_payload


BENCHMARKS = {
    "payload_build_50k_ms": 30.0,
    "window_slice_50k_ms": 12.0,
    "adaptive_coalesce_10_us": 250.0,
}


def _events(count=50000):
    indices = np.arange(count, dtype=np.int64)
    events = np.empty(count, dtype=EVENT_CD_DTYPE)
    events["x"] = indices % 640
    events["y"] = (indices // 640) % 480
    events["p"] = indices % 2
    events["t"] = indices * 10
    return events


def _median_duration(operation, repeats, scale):
    operation()
    samples = []
    for _unused in range(repeats):
        started_at = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started_at) * scale)
    return statistics.median(samples)


def run_benchmarks(repeats=7):
    events = _events()
    rng = np.random.default_rng(12345)

    def build_payload():
        payload = build_inference_payload(
            events,
            width=640,
            height=480,
            fallback_normalization="full",
            rng=rng,
        )
        if payload is None:
            raise RuntimeError("payload benchmark produced no payload")

    def slice_windows():
        chunks = EventWindowSlicer(20000).consume(events)
        if len(chunks) != 24:
            raise RuntimeError("window benchmark produced unexpected chunks")

    sample = events[:1]
    stop_signal = object()

    def coalesce_queue():
        for batch in range(200):
            base = float(batch)
            target = queue.Queue(maxsize=10)
            first = InferenceWindow(sample, None, 0, ready_at=base + 0.001)
            for index in range(10):
                target.put_nowait(
                    InferenceWindow(
                        sample,
                        None,
                        0,
                        ready_at=base + 0.002 + index * 0.001,
                    )
                )
            consumer = AdaptiveInferenceQueueConsumer(
                target,
                stop_signal,
                clock=lambda: base + 0.02,
            )
            consumer.select(first)
            if consumer.dropped_windows != 10:
                raise RuntimeError("adaptive queue benchmark did not coalesce")

    return {
        "payload_build_50k_ms": _median_duration(build_payload, repeats, 1000.0),
        "window_slice_50k_ms": _median_duration(slice_windows, repeats, 1000.0),
        "adaptive_coalesce_10_us": (
            _median_duration(coalesce_queue, repeats, 1000000.0) / 200.0
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)

    results = run_benchmarks(repeats=max(3, args.repeats))
    print(json.dumps(results, indent=2, sort_keys=True))
    if not args.check:
        return 0

    failures = [
        "%s %.2f exceeded %.2f" % (name, results[name], budget)
        for name, budget in BENCHMARKS.items()
        if results[name] > budget
    ]
    if failures:
        raise SystemExit("Performance regression detected:\n" + "\n".join(failures))
    print("Performance regression budgets passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
