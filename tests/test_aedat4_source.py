import queue

import numpy as np

from backend.aedat4_source import run_aedat4_replay_loop
from backend.event_processing import EVENT_CD_DTYPE


class FakeEventBatch:
    def __init__(self, events):
        self._events = events

    def isEmpty(self):
        return len(self._events) == 0

    def numpy(self):
        return self._events


class FakeReader:
    def __init__(self, batches):
        self.batches = list(batches)

    def isRunning(self):
        return bool(self.batches)

    def getNextEventBatch(self):
        if not self.batches:
            return None
        return self.batches.pop(0)


class FakeFrameGenerator:
    def __init__(self):
        self.frames = []

    def process_events(self, events):
        self.frames.append(events)


class PassthroughFilter:
    def apply(self, events):
        return events


def test_aedat4_renderer_emits_first_events_immediately():
    events = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 40000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    frame_generator = FakeFrameGenerator()

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(events)]),
        frame_generator=frame_generator,
        fps=30,
        nn_interval_us=1000,
        is_running=lambda: True,
        roi_getter=lambda: None,
        nn_queue=queue.Queue(),
        noise_filter=PassthroughFilter(),
        sleep=lambda _: None,
        now=lambda: 0.0,
    )

    assert len(frame_generator.frames) == 1
    assert frame_generator.frames[0].dtype == EVENT_CD_DTYPE
    assert frame_generator.frames[0].tolist() == [(1, 2, 1, 100), (3, 4, 0, 40000)]


def test_aedat4_renderer_replays_later_events_by_frame_time():
    first = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 40000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    second = np.array(
        [(5, 6, 1, 50000), (7, 8, 0, 80000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    frame_generator = FakeFrameGenerator()
    sleep_calls = []

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(first), FakeEventBatch(second)]),
        frame_generator=frame_generator,
        fps=30,
        nn_interval_us=100000,
        is_running=lambda: True,
        roi_getter=lambda: None,
        nn_queue=queue.Queue(),
        noise_filter=PassthroughFilter(),
        sleep=lambda seconds: sleep_calls.append(seconds),
        now=lambda: 0.0,
    )

    assert len(frame_generator.frames) == 3
    assert frame_generator.frames[0].tolist() == [(1, 2, 1, 100), (3, 4, 0, 40000)]
    assert frame_generator.frames[1].tolist() == [(5, 6, 1, 50000)]
    assert frame_generator.frames[2].tolist() == [(7, 8, 0, 80000)]
    assert sleep_calls
    assert sleep_calls[0] > 0


def test_aedat4_renderer_applies_replay_factor_to_frame_sleep():
    first = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 40000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    second = np.array(
        [(5, 6, 1, 50000), (7, 8, 0, 80000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    sleep_calls = []

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(first), FakeEventBatch(second)]),
        frame_generator=FakeFrameGenerator(),
        fps=30,
        nn_interval_us=100000,
        is_running=lambda: True,
        roi_getter=lambda: None,
        nn_queue=queue.Queue(),
        noise_filter=PassthroughFilter(),
        sleep=lambda seconds: sleep_calls.append(seconds),
        now=lambda: 0.0,
        replay_factor=2.0,
    )

    assert sleep_calls[0] < 0.02


def test_aedat4_replay_replaces_full_nn_queue_without_blocking():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 200), (3, 3, 1, 300)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    nn_queue = queue.Queue(maxsize=1)
    nn_queue.put_nowait("old")

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(events)]),
        frame_generator=FakeFrameGenerator(),
        fps=30,
        nn_interval_us=50,
        is_running=lambda: True,
        roi_getter=lambda: None,
        nn_queue=nn_queue,
        noise_filter=PassthroughFilter(),
        sleep=lambda _: None,
        now=lambda: 0.0,
    )

    queued = nn_queue.get_nowait()
    assert isinstance(queued, np.ndarray)
    assert queued.dtype == EVENT_CD_DTYPE
