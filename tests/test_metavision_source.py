import queue

import numpy as np
import pytest

from backend.event_processing import EVENT_CD_DTYPE
from backend.metavision_source import DynamicReplayEventsIterator, metavision_replay_factor, run_metavision_event_loop


class PassthroughFilter:
    def apply(self, events):
        return events


class FrameGenerator:
    def __init__(self):
        self.frames = []

    def process_events(self, events):
        self.frames.append(events)


class FakeMetavisionIterator:
    delta_t = 20000

    def __init__(self, timestamps, start_ts=0):
        self.timestamps = timestamps
        self.start_ts = start_ts
        self.current_time = 0

    def __iter__(self):
        for timestamp in self.timestamps:
            self.current_time = timestamp
            yield np.array([(timestamp,)], dtype=[("t", "<i8")])

    def get_current_time(self):
        return self.current_time

    def get_size(self):
        return 640, 480


def test_run_metavision_event_loop_filters_roi_and_replaces_queue():
    events = np.array(
        [(1, 1, 1, 100), (10, 10, 1, 110), (11, 11, 0, 120)],
        dtype=EVENT_CD_DTYPE,
    )
    target_queue = queue.Queue(maxsize=1)
    target_queue.put_nowait("old")
    frame_generator = FrameGenerator()

    run_metavision_event_loop(
        iterator=[events],
        is_running=lambda: True,
        roi_getter=lambda: (10, 10, 5, 5),
        noise_filter=PassthroughFilter(),
        frame_generator=frame_generator,
        nn_queue=target_queue,
    )

    queued = target_queue.get_nowait()
    assert queued.tolist() == [(10, 10, 1, 110), (11, 11, 0, 120)]
    assert frame_generator.frames[0].tolist() == queued.tolist()


def test_run_metavision_event_loop_slices_nn_events_independently_from_display():
    events = np.array(
        [
            (1, 1, 1, 0),
            (2, 2, 1, 10000),
            (3, 3, 0, 21000),
            (4, 4, 0, 30000),
            (5, 5, 1, 41000),
        ],
        dtype=EVENT_CD_DTYPE,
    )
    target_queue = queue.Queue(maxsize=10)
    frame_generator = FrameGenerator()

    run_metavision_event_loop(
        iterator=[events],
        is_running=lambda: True,
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        frame_generator=frame_generator,
        nn_queue=target_queue,
        nn_interval_us=20000,
    )

    assert frame_generator.frames[0].tolist() == events.tolist()
    chunks = []
    while not target_queue.empty():
        chunks.append(target_queue.get_nowait())
    assert [chunk["t"].tolist() for chunk in chunks] == [[0, 10000], [21000, 30000]]


def test_run_metavision_event_loop_reports_progress_before_filters():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 0, 200)],
        dtype=EVENT_CD_DTYPE,
    )
    progress = []

    run_metavision_event_loop(
        iterator=[events],
        is_running=lambda: True,
        roi_getter=lambda: (99, 99, 1, 1),
        noise_filter=PassthroughFilter(),
        frame_generator=FrameGenerator(),
        nn_queue=queue.Queue(),
        progress_callback=lambda timestamp: progress.append(timestamp),
    )

    assert progress == [200]


def test_metavision_replay_factor_converts_speed_to_time_scale():
    assert metavision_replay_factor(4.0) == pytest.approx(0.25)
    assert metavision_replay_factor(0.25) == pytest.approx(4.0)
    assert metavision_replay_factor(1.0) == pytest.approx(1.0)


def test_dynamic_replay_iterator_applies_speed_factor():
    current_time = [0.0]
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration

    iterator = DynamicReplayEventsIterator(
        FakeMetavisionIterator([20000, 40000]),
        replay_factor_getter=lambda: 2.0,
        sleep=fake_sleep,
        now=lambda: current_time[0],
    )

    list(iterator)

    assert sleeps == [pytest.approx(0.01), pytest.approx(0.01)]


def test_dynamic_replay_iterator_anchors_sleep_to_start_ts_after_seek():
    current_time = [0.0]
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration

    iterator = DynamicReplayEventsIterator(
        FakeMetavisionIterator([4020000], start_ts=4000000),
        replay_factor_getter=lambda: 1.0,
        sleep=fake_sleep,
        now=lambda: current_time[0],
    )

    list(iterator)

    assert sleeps == [pytest.approx(0.02)]


def test_dynamic_replay_iterator_reanchors_when_speed_changes_during_sleep():
    current_time = [0.0]
    replay_factor = [0.25]
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration
        replay_factor[0] = 4.0

    iterator = DynamicReplayEventsIterator(
        FakeMetavisionIterator([20000]),
        replay_factor_getter=lambda: replay_factor[0],
        sleep=fake_sleep,
        now=lambda: current_time[0],
    )

    list(iterator)

    assert sleeps == [pytest.approx(0.05)]
