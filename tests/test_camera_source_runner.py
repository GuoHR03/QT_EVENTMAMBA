from types import SimpleNamespace

import pytest

from backend.camera_source_factory import SOURCE_H5, SOURCE_METAVISION
from backend.camera_source_runner import CameraRunContext, close_camera_source, run_camera_source


class ClosableFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class NoiseFilter:
    def apply(self, events):
        return events


def test_close_camera_source_closes_file_when_present():
    source_file = ClosableFile()
    close_camera_source(SimpleNamespace(file=source_file))

    assert source_file.closed


def test_close_camera_source_ignores_sources_without_file():
    close_camera_source(SimpleNamespace())


def test_run_camera_source_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported camera source type"):
        run_camera_source("unknown", SimpleNamespace(), _context())


def test_run_h5_source_runs_replay_and_closes_file(monkeypatch):
    calls = {}
    source_file = ClosableFile()
    source = SimpleNamespace(
        file=source_file,
        events_dataset=[1, 2, 3],
        dtype_names=("x", "y", "p", "t"),
        width=640,
        height=480,
        frame_generator=object(),
    )

    class FakeProcessor:
        def __init__(self, **kwargs):
            calls["processor"] = kwargs
            self.handle_frame_events = lambda events: None
            calls["processor_instance"] = self

    def fake_replay_loop(**kwargs):
        calls["replay"] = kwargs

    monkeypatch.setattr("backend.camera_source_runner.H5FrameProcessor", FakeProcessor)
    monkeypatch.setattr("backend.camera_source_runner.run_h5_replay_loop", fake_replay_loop)

    run_camera_source(SOURCE_H5, source, _context())

    assert source_file.closed
    assert calls["processor"]["width"] == 640
    assert calls["processor"]["height"] == 480
    assert calls["replay"]["events_dataset"] == [1, 2, 3]
    assert calls["replay"]["dtype_names"] == ("x", "y", "p", "t")
    assert calls["replay"]["handle_frame_events"] == calls["processor_instance"].handle_frame_events
    assert callable(calls["replay"]["now"])
    assert callable(calls["replay"]["sleep"])
    assert calls["replay"]["replay_factor_getter"]() == 1.5
    assert calls["replay"]["fps_getter"]() == 30


def test_run_metavision_source_passes_nn_interval(monkeypatch):
    calls = {}
    source = SimpleNamespace(
        iterator=object(),
        frame_generator=object(),
    )

    def fake_event_loop(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("backend.camera_source_runner.run_metavision_event_loop", fake_event_loop)

    run_camera_source(SOURCE_METAVISION, source, _context())

    assert calls["iterator"] is source.iterator
    assert calls["frame_generator"] is source.frame_generator
    assert calls["nn_interval_us"] == 20000


def _context():
    return CameraRunContext(
        fps=30,
        fps_getter=lambda: 30,
        nn_interval_us=20000,
        replay_factor=1.5,
        replay_factor_getter=lambda: 1.5,
        is_running=lambda: True,
        roi_getter=lambda: None,
        image_callback=lambda image, timestamp: None,
        nn_queue=None,
        noise_filter=NoiseFilter(),
        target_queue=None,
        analysis_enabled=lambda: True,
    )
