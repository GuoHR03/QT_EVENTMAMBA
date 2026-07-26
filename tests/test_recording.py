from pathlib import Path

import backend.recording as recording_module
from backend.recording import RawRecorder


class FakeEventsStream:
    def __init__(self):
        self.started_path = None
        self.stop_count = 0

    def start_log_raw_data(self, path):
        self.started_path = path

    def stop_log_raw_data(self):
        self.stop_count += 1


class FakeDevice:
    def __init__(self, stream):
        self.stream = stream

    def get_i_events_stream(self):
        return self.stream


def test_raw_recorder_starts_with_timestamped_filename():
    stream = FakeEventsStream()
    recorder = RawRecorder(clock=lambda _: "20260623_120000")

    started = recorder.start(FakeDevice(stream))

    assert started
    assert recorder.is_recording
    assert stream.started_path == "recording_20260623_120000.raw"


def test_frozen_raw_recorder_uses_writable_user_directory(tmp_path, monkeypatch):
    stream = FakeEventsStream()
    monkeypatch.setattr(recording_module.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    recorder = RawRecorder(clock=lambda _: "20260623_120000")

    assert recorder.start(FakeDevice(stream))

    expected_path = (
        tmp_path
        / "LocalAppData"
        / "UI_Event"
        / "record"
        / "recording_20260623_120000.raw"
    )
    assert Path(stream.started_path) == expected_path
    assert expected_path.parent.is_dir()


def test_raw_recorder_stops_stream():
    stream = FakeEventsStream()
    recorder = RawRecorder(clock=lambda _: "20260623_120000")
    recorder.start(FakeDevice(stream))

    stopped = recorder.stop(FakeDevice(stream))

    assert stopped
    assert not recorder.is_recording
    assert stream.stop_count == 1


def test_raw_recorder_handles_missing_device_or_stream():
    recorder = RawRecorder(clock=lambda _: "20260623_120000")

    assert not recorder.start(None)
    assert not recorder.is_recording
    assert not recorder.stop(FakeDevice(None))
    assert not recorder.is_recording
