import queue

import numpy as np

from backend.aedat4_source import run_aedat4_replay_loop
from backend.event_pipeline import EventPipeline
from backend.event_processing import EVENT_CD_DTYPE


class FakeEventBatch:
    def __init__(self, events):
        self._events = events

    def isEmpty(self):
        return len(self._events) == 0

    def numpy(self):
        return self._events


class CallbackEventBatch(FakeEventBatch):
    def __init__(self, events, callback):
        super().__init__(events)
        self._callback = callback

    def numpy(self):
        self._callback()
        return super().numpy()


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


def make_pipeline(renderer, nn_queue, interval_us, roi_getter=lambda: None):
    return EventPipeline(
        roi_getter=roi_getter,
        noise_filter=PassthroughFilter(),
        renderer=renderer,
        inference_queue=nn_queue,
        inference_interval_us=interval_us,
    )


def test_aedat4_renderer_splits_first_batch_at_frame_boundary():
    events = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 40000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    frame_generator = FakeFrameGenerator()
    sleep_calls = []

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(events)]),
        event_pipeline=make_pipeline(frame_generator, queue.Queue(), 1000),
        fps=30,
        is_running=lambda: True,
        sleep=lambda seconds: sleep_calls.append(seconds),
        now=lambda: 0.0,
    )

    assert len(frame_generator.frames) == 2
    assert frame_generator.frames[0].dtype == EVENT_CD_DTYPE
    assert frame_generator.frames[0].tolist() == [(1, 2, 1, 100)]
    assert frame_generator.frames[1].tolist() == [(3, 4, 0, 40000)]
    assert sleep_calls[0] > 0


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
        event_pipeline=make_pipeline(frame_generator, queue.Queue(), 100000),
        fps=30,
        is_running=lambda: True,
        sleep=lambda seconds: sleep_calls.append(seconds),
        now=lambda: 0.0,
    )

    assert len(frame_generator.frames) == 3
    assert frame_generator.frames[0].tolist() == [(1, 2, 1, 100)]
    assert frame_generator.frames[1].tolist() == [(3, 4, 0, 40000), (5, 6, 1, 50000)]
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
        event_pipeline=make_pipeline(FakeFrameGenerator(), queue.Queue(), 100000),
        fps=30,
        is_running=lambda: True,
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
        event_pipeline=make_pipeline(FakeFrameGenerator(), nn_queue, 50),
        fps=30,
        is_running=lambda: True,
        sleep=lambda _: None,
        now=lambda: 0.0,
    )

    queued = nn_queue.get_nowait()
    assert isinstance(queued, np.ndarray)
    assert queued.dtype == EVENT_CD_DTYPE


def test_aedat4_replay_skips_events_before_start_time_and_reports_progress():
    first = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 40000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    second = np.array(
        [(5, 6, 1, 50000), (7, 8, 0, 80000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    frame_generator = FakeFrameGenerator()
    progress = []

    run_aedat4_replay_loop(
        reader=FakeReader([FakeEventBatch(first), FakeEventBatch(second)]),
        event_pipeline=make_pipeline(frame_generator, queue.Queue(), 100000),
        fps=30,
        is_running=lambda: True,
        sleep=lambda _: None,
        now=lambda: 0.0,
        start_time_us=45000,
        progress_callback=lambda timestamp: progress.append(timestamp),
    )

    assert frame_generator.frames[0].tolist() == [(5, 6, 1, 50000), (7, 8, 0, 80000)]
    assert progress == [80000]


def test_aedat4_roi_generation_change_discards_buffered_old_roi_events():
    first = np.array(
        [(1, 1, 1, 100)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    second = np.array(
        [(21, 1, 1, 200)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    roi_state = {"generation": 0, "roi": (0, 0, 10, 10)}

    def switch_roi():
        roi_state.update(generation=1, roi=(20, 0, 10, 10))

    renderer = FakeFrameGenerator()
    pipeline = EventPipeline(
        roi_getter=lambda: roi_state["roi"],
        roi_snapshot_getter=lambda: (
            roi_state["generation"],
            roi_state["roi"],
        ),
        noise_filter=PassthroughFilter(),
        renderer=renderer,
    )

    run_aedat4_replay_loop(
        reader=FakeReader(
            [
                FakeEventBatch(first),
                CallbackEventBatch(second, switch_roi),
            ]
        ),
        event_pipeline=pipeline,
        fps=30,
        is_running=lambda: True,
        sleep=lambda _seconds: None,
        now=lambda: 0.0,
    )

    assert len(renderer.frames) == 1
    assert renderer.frames[0].tolist() == [(21, 1, 1, 200)]


def test_aedat4_inference_generation_change_keeps_same_roi_display_buffer():
    first = np.array(
        [(1, 1, 1, 100)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    second = np.array(
        [(2, 2, 1, 200)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    snapshot = {"generation": 0, "roi": None}

    def change_inference_generation():
        snapshot["generation"] += 1

    renderer = FakeFrameGenerator()
    pipeline = EventPipeline(
        roi_getter=lambda: snapshot["roi"],
        roi_snapshot_getter=lambda: (
            snapshot["generation"],
            snapshot["roi"],
        ),
        noise_filter=PassthroughFilter(),
        renderer=renderer,
        analysis_enabled=lambda: False,
    )

    run_aedat4_replay_loop(
        reader=FakeReader(
            [
                FakeEventBatch(first),
                CallbackEventBatch(second, change_inference_generation),
            ]
        ),
        event_pipeline=pipeline,
        fps=30,
        is_running=lambda: True,
        sleep=lambda _seconds: None,
        now=lambda: 0.0,
    )

    assert len(renderer.frames) == 1
    assert renderer.frames[0].tolist() == [
        (1, 1, 1, 100),
        (2, 2, 1, 200),
    ]
