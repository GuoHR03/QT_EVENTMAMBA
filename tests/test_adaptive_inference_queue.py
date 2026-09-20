import queue

import numpy as np
import pytest

from backend.adaptive_inference_queue import AdaptiveInferenceQueueConsumer
from backend.event_pipeline import InferenceWindow
from backend.event_processing import EVENT_CD_DTYPE


STOP = object()


def _window(timestamp, ready_at):
    events = np.array([(1, 1, 1, timestamp)], dtype=EVENT_CD_DTYPE)
    return InferenceWindow(events, None, 0, ready_at=ready_at)


def test_healthy_queue_keeps_fifo_order():
    target = queue.Queue(maxsize=10)
    second = _window(2, 10.02)
    target.put_nowait(second)
    consumer = AdaptiveInferenceQueueConsumer(target, STOP, clock=lambda: 10.03)

    first = _window(1, 10.0)

    assert consumer.select(first) is first
    assert target.get_nowait() is second
    assert consumer.dropped_windows == 0


def test_backlog_is_coalesced_to_the_freshest_window():
    target = queue.Queue(maxsize=10)
    queued = [
        _window(index, 10.0 + (index - 1) * 0.02)
        for index in range(2, 5)
    ]
    for item in queued:
        target.put_nowait(item)
    consumer = AdaptiveInferenceQueueConsumer(target, STOP, clock=lambda: 10.09)

    selected = consumer.select(_window(1, 10.0))

    assert selected is queued[-1]
    assert target.empty()
    assert consumer.dropped_windows == 3
    assert consumer.arrival_ewma_s == pytest.approx(0.02)


def test_slow_processing_coalesces_even_one_pending_window():
    target = queue.Queue(maxsize=10)
    consumer = AdaptiveInferenceQueueConsumer(target, STOP, clock=lambda: 20.03)
    consumer.select(_window(1, 20.0))
    consumer.observe_processing(0.03)
    second = _window(2, 20.02)
    assert consumer.select(second) is second
    consumer.observe_processing(0.03)
    latest = _window(4, 20.06)
    target.put_nowait(latest)

    selected = consumer.select(_window(3, 20.04))

    assert selected is latest
    assert consumer.dropped_windows == 1


def test_old_window_is_coalesced_using_adaptive_freshness_budget():
    target = queue.Queue(maxsize=10)
    latest = _window(2, 30.09)
    target.put_nowait(latest)
    consumer = AdaptiveInferenceQueueConsumer(target, STOP, clock=lambda: 30.2)

    assert consumer.select(_window(1, 30.0)) is latest
    assert consumer.freshness_budget_s >= 0.05


def test_stop_signal_survives_backlog_coalescing():
    target = queue.Queue(maxsize=4)
    latest = _window(3, 40.04)
    target.put_nowait(_window(2, 40.02))
    target.put_nowait(latest)
    target.put_nowait(STOP)
    consumer = AdaptiveInferenceQueueConsumer(target, STOP, clock=lambda: 40.05)

    assert consumer.select(_window(1, 40.0)) is latest
    assert target.get_nowait() is STOP


def test_stop_signal_is_selected_immediately():
    consumer = AdaptiveInferenceQueueConsumer(queue.Queue(), STOP)

    assert consumer.select(STOP) is STOP
    assert consumer.selected_windows == 0
