import queue
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from backend.event_pipeline import EventPipeline
from backend.event_processing import EVENT_CD_DTYPE
from backend.metavision_source import (
    DynamicReplayEventsIterator,
    apply_hardware_roi,
    run_metavision_event_loop,
)


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


class FakeGeometry:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height


class FakeHardwareRoi:
    def __init__(self, set_window_result=True, enable_result=True):
        self.set_window_result = set_window_result
        self.enable_result = enable_result
        self.windows = []
        self.enabled = []

    def set_window(self, window):
        self.windows.append(window)
        return self.set_window_result

    def enable(self, enabled):
        self.enabled.append(enabled)
        return self.enable_result


class FakeDevice:
    def __init__(self, geometry=None, i_roi=None):
        self.geometry = geometry
        self.i_roi = i_roi
        self.get_i_roi_calls = 0

    def get_i_geometry(self):
        return self.geometry

    def get_i_roi(self):
        self.get_i_roi_calls += 1
        return self.i_roi


def install_fake_metavision_hal(monkeypatch):
    window_calls = []
    fake_hal = ModuleType("libs.metavision_hal")

    def make_window(*args):
        window_calls.append(args)
        return SimpleNamespace(args=args)

    fake_hal.I_ROI = SimpleNamespace(Window=make_window)
    fake_libs = ModuleType("libs")
    fake_libs.metavision_hal = fake_hal
    monkeypatch.setitem(sys.modules, "libs", fake_libs)
    monkeypatch.setitem(sys.modules, "libs.metavision_hal", fake_hal)
    return window_calls


def test_apply_hardware_roi_clips_to_1280x720_and_passes_width_height(monkeypatch):
    window_calls = install_fake_metavision_hal(monkeypatch)
    hardware_roi = FakeHardwareRoi()
    device = FakeDevice(FakeGeometry(), hardware_roi)

    applied = apply_hardware_roi(device, (-20, 700, 100, 50))

    assert applied == (0, 700, 80, 20)
    assert window_calls == [(0, 700, 80, 20)]
    assert hardware_roi.windows[0].args == (0, 700, 80, 20)
    assert hardware_roi.enabled == [True]


@pytest.mark.parametrize(
    ("set_window_result", "enable_result", "error"),
    [
        (False, True, "rejected the hardware ROI window"),
        (True, False, "failed to enable hardware ROI"),
    ],
)
def test_apply_hardware_roi_raises_when_hal_rejects_operation(
    monkeypatch,
    set_window_result,
    enable_result,
    error,
):
    install_fake_metavision_hal(monkeypatch)
    hardware_roi = FakeHardwareRoi(set_window_result, enable_result)
    device = FakeDevice(FakeGeometry(), hardware_roi)

    with pytest.raises(RuntimeError, match=error):
        apply_hardware_roi(device, (100, 200, 300, 250))

    assert len(hardware_roi.windows) == 1
    assert hardware_roi.enabled == ([True] if set_window_result else [])


def test_apply_hardware_roi_skips_device_without_geometry():
    hardware_roi = FakeHardwareRoi()
    device = FakeDevice(geometry=None, i_roi=hardware_roi)

    assert apply_hardware_roi(device, (10, 20, 30, 40)) is None
    assert device.get_i_roi_calls == 0
    assert hardware_roi.windows == []
    assert hardware_roi.enabled == []


def test_apply_hardware_roi_skips_fully_outside_sensor():
    hardware_roi = FakeHardwareRoi()
    device = FakeDevice(FakeGeometry(), hardware_roi)

    assert apply_hardware_roi(device, (1280, 100, 20, 20)) is None
    assert device.get_i_roi_calls == 0
    assert hardware_roi.windows == []
    assert hardware_roi.enabled == []


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
        event_pipeline=EventPipeline(
            roi_getter=lambda: (10, 10, 5, 5),
            noise_filter=PassthroughFilter(),
            renderer=frame_generator,
            inference_queue=target_queue,
        ),
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
        event_pipeline=EventPipeline(
            roi_getter=lambda: None,
            noise_filter=PassthroughFilter(),
            renderer=frame_generator,
            inference_queue=target_queue,
            inference_interval_us=20000,
        ),
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
        event_pipeline=EventPipeline(
            roi_getter=lambda: (99, 99, 1, 1),
            noise_filter=PassthroughFilter(),
            renderer=FrameGenerator(),
            inference_queue=queue.Queue(),
        ),
        progress_callback=lambda timestamp: progress.append(timestamp),
    )

    assert progress == [200]


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


def test_dynamic_replay_iterator_stops_during_replay_wait():
    current_time = [0.0]
    iterator = None

    def fake_sleep(_duration):
        iterator.request_stop()

    iterator = DynamicReplayEventsIterator(
        FakeMetavisionIterator([20000]),
        replay_factor_getter=lambda: 1.0,
        sleep=fake_sleep,
        now=lambda: current_time[0],
    )

    assert list(iterator) == []
