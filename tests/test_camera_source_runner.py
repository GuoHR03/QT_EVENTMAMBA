from types import SimpleNamespace

import pytest

from backend.camera_source_runner import CameraRunContext, close_camera_source, run_camera_source


class NoiseFilter:
    def apply(self, events):
        return events


class FakeSource:
    def __init__(self, error=None):
        self.renderer = object()
        self.error = error
        self.run_calls = []
        self.close_count = 0

    def run(self, context, event_pipeline):
        self.run_calls.append((context, event_pipeline))
        if self.error is not None:
            raise self.error

    def close(self):
        self.close_count += 1


def test_run_camera_source_uses_common_source_interface():
    source = FakeSource()
    context = _context()

    run_camera_source(source, context)

    assert source.run_calls[0][0] is context
    assert source.run_calls[0][1].renderer is source.renderer
    assert source.run_calls[0][1].inference_slicer.interval_us == 20000
    assert source.close_count == 0


def test_run_camera_source_propagates_run_error_to_session_owner():
    source = FakeSource(RuntimeError("failed"))

    with pytest.raises(RuntimeError, match="failed"):
        run_camera_source(source, _context())

    assert source.close_count == 0


def test_close_camera_source_uses_legacy_file_fallback():
    source_file = SimpleNamespace(closed=False)
    source_file.close = lambda: setattr(source_file, "closed", True)

    close_camera_source(SimpleNamespace(file=source_file))

    assert source_file.closed


def test_run_camera_source_ignores_none():
    run_camera_source(None, _context())


def _context():
    return CameraRunContext(
        fps=30,
        fps_getter=lambda: 30,
        nn_interval_us=20000,
        replay_factor=1.5,
        replay_factor_getter=lambda: 1.5,
        is_running=lambda: True,
        roi_getter=lambda: None,
        nn_queue=None,
        noise_filter=NoiseFilter(),
        analysis_enabled=lambda: True,
    )
