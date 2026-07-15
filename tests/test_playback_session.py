import pytest

from backend.camera_source_runner import CameraRunContext
from backend.event_source import SourceMetadata
from backend.playback_session import PlaybackSession


class NoiseFilter:
    def __init__(self):
        self.initialized = []

    def initialize(self, width, height):
        self.initialized.append((width, height))

    def apply(self, events):
        return events


class Source:
    def __init__(self, error=None):
        self.renderer = object()
        self.error = error
        self.run_count = 0
        self.running_states = []
        self.close_count = 0
        self.request_stop_count = 0
        self._metadata = SourceMetadata("h5", 320, 240)

    def metadata(self):
        return self._metadata

    def run(self, context, event_pipeline):
        self.run_count += 1
        self.running_states.append(context.is_running())
        if self.error is not None:
            raise self.error

    def close(self):
        self.close_count += 1

    def request_stop(self):
        self.request_stop_count += 1


class Worker:
    def __init__(self):
        self.start_count = 0
        self.stop_count = 0
        self.stop_modes = []
        self.wait_count = 0

    def start(self):
        self.start_count += 1

    def stop(self, discard_pending=True):
        self.stop_count += 1
        self.stop_modes.append(discard_pending)

    def wait(self):
        self.wait_count += 1


def test_playback_session_owns_source_filter_and_worker_lifecycle():
    source = Source()
    worker = Worker()
    context = _context()
    session = PlaybackSession(source, context, worker)

    session.run()

    assert source.run_count == 1
    assert source.running_states == [True]
    assert source.close_count == 1
    assert context.noise_filter.initialized == [(320, 240)]
    assert (worker.start_count, worker.stop_count, worker.wait_count) == (1, 1, 1)
    assert worker.stop_modes == [False]
    assert not session.is_running()


def test_playback_session_cleans_up_after_source_error():
    source = Source(RuntimeError("failed"))
    worker = Worker()
    session = PlaybackSession(source, _context(), worker)

    with pytest.raises(RuntimeError, match="failed"):
        session.run()

    assert source.close_count == 1
    assert (worker.stop_count, worker.wait_count) == (1, 1)
    assert worker.stop_modes == [True]


def test_playback_session_stop_before_run_skips_source_and_worker():
    source = Source()
    worker = Worker()
    session = PlaybackSession(source, _context(), worker)

    session.stop()
    session.run()

    assert source.run_count == 0
    assert source.request_stop_count == 1
    assert source.close_count == 1
    assert worker.start_count == 0
    assert worker.stop_count == 1
    assert worker.stop_modes == [True]
    assert worker.wait_count == 0


def _context():
    return CameraRunContext(
        fps=30,
        fps_getter=lambda: 30,
        nn_interval_us=20000,
        replay_factor=1.0,
        replay_factor_getter=lambda: 1.0,
        is_running=lambda: False,
        roi_getter=lambda: None,
        nn_queue=None,
        noise_filter=NoiseFilter(),
        analysis_enabled=lambda: True,
    )
