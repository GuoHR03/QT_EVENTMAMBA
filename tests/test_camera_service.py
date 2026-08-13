import queue

import pytest

from backend.camera_service import CameraService
from backend.camera_service import SOURCE_MODE_FILE, SOURCE_MODE_LIVE


class FakeSignal:
    def __init__(self):
        self.callbacks = []
        self.emissions = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, *args):
        self.emissions.append(args)
        for callback in tuple(self.callbacks):
            callback(*args)


class FakeCameraThread:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.image_signal = FakeSignal()
        self.status_signal = FakeSignal()
        self.finished_signal = FakeSignal()
        self.progress_signal = FakeSignal()
        self.source_ready_signal = FakeSignal()
        self.is_recording = False
        self.width = 640
        self.height = 480
        self.started = False
        self.stopped = False
        self.deleted = False
        self.recording_start_result = True
        self.recording_stop_result = True
        self.recording_start_count = 0
        self.recording_stop_count = 0
        self.wait_results = []
        self.interruption_requested = False
        self.terminated = False
        self.analysis_enabled_calls = []

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started and not self.stopped

    def stop(self):
        self.stopped = True

    def wait(self, _timeout=None):
        if self.wait_results:
            return self.wait_results.pop(0)
        return True

    def requestInterruption(self):
        self.interruption_requested = True

    def terminate(self):
        self.terminated = True

    def deleteLater(self):
        self.deleted = True

    def start_recording(self):
        self.recording_start_count += 1
        self.is_recording = self.recording_start_result
        return self.recording_start_result

    def stop_recording(self):
        self.recording_stop_count += 1
        if self.recording_stop_result:
            self.is_recording = False
        return self.recording_stop_result

    def set_analysis_enabled(self, enabled):
        self.analysis_enabled_calls.append(bool(enabled))
        return True


class FakeMetadataService:
    def __init__(self):
        self.recorded = []
        self.scan_requests = []
        self.callback = None

    def duration_hint(self, input_path):
        return 9000 if input_path else 0

    def record_duration(self, input_path, duration_us):
        self.recorded.append((input_path, duration_us))

    def ensure_duration_scan(self, input_path, callback=None):
        self.scan_requests.append(input_path)
        self.callback = callback
        return True


def _service(metadata_service=None, frame_queue=None, source_ready_callback=None):
    created_threads = []

    def thread_factory(**kwargs):
        thread = FakeCameraThread(**kwargs)
        created_threads.append(thread)
        return thread

    signals = [FakeSignal() for _ in range(4)]
    service = CameraService(
        frame_queue=frame_queue or queue.Queue(),
        image_signal=signals[0],
        status_signal=signals[1],
        finished_signal=signals[2],
        progress_signal=signals[3],
        thread_factory=thread_factory,
        metadata_service=metadata_service or FakeMetadataService(),
        source_ready_callback=source_ready_callback,
    )
    return service, created_threads, signals


def test_camera_service_injects_thread_and_metadata_dependencies():
    metadata = FakeMetadataService()
    service, threads, signals = _service(metadata)
    service.set_input_file("events.raw")

    service.start()

    thread = threads[0]
    assert thread.started
    assert thread.kwargs["file_path"] == "events.raw"
    assert thread.kwargs["duration_hint_us"] == 9000
    assert metadata.scan_requests == ["events.raw"]
    assert service.current_size() is None
    thread.source_ready_signal.emit(640, 480)
    assert service.current_size() == (640, 480)

    thread.progress_signal.emit(1200, 9000)
    assert metadata.recorded == [("events.raw", 9000)]
    assert signals[3].emissions[-1] == (1200, 9000)

    service.stop()
    assert thread.stopped
    assert thread.deleted
    assert service.thread is None


def test_camera_service_persists_and_forwards_inference_generation_gate():
    service, threads, _signals = _service()

    assert service.set_analysis_enabled(True)
    service.start()
    assert threads[0].kwargs["analysis_enabled"] is True

    assert service.set_analysis_enabled(False)
    assert threads[0].analysis_enabled_calls == [False]

    service.restart()
    assert threads[1].kwargs["analysis_enabled"] is False


def test_camera_service_ignores_old_file_duration_callback():
    metadata = FakeMetadataService()
    service, _threads, signals = _service(metadata)
    service.set_input_file("current.raw")
    service._last_progress_current_us = 7000

    service._handle_duration_resolved("old.raw", 10000)
    assert signals[3].emissions == [(0, 0)]

    service._handle_duration_resolved("current.raw", 5000)
    assert signals[3].emissions == [(0, 0), (5000, 5000)]


def test_camera_service_switches_running_file_source_back_to_live_and_resets_state():
    frame_queue = queue.Queue()
    service, threads, signals = _service(frame_queue=frame_queue)
    service.set_input_file("events.raw")
    service.start(seek_fraction=0.75)
    frame_queue.put_nowait({"msg_type": "EVENTS"})

    restarted = service.set_live_camera()

    assert restarted
    assert service.source_mode == SOURCE_MODE_LIVE
    assert service.file_path is None
    assert service.last_seek_fraction == 0.0
    assert service._last_progress_current_us == 0
    assert frame_queue.empty()
    assert threads[0].stopped
    assert threads[0].deleted
    assert threads[1].started
    assert "file_path" not in threads[1].kwargs
    assert threads[1].kwargs["seek_fraction"] == 0.0
    assert signals[3].emissions[-1] == (0, 0)


def test_camera_service_switches_to_raw_file_without_restarting_when_requested():
    service, threads, _signals = _service()
    service.start()

    restarted = service.set_input_file(
        "recording.raw",
        restart_if_running=False,
    )

    assert not restarted
    assert service.source_mode == SOURCE_MODE_FILE
    assert service.file_path == "recording.raw"
    assert threads[0].stopped
    assert service.thread is None


def test_camera_service_propagates_recording_results_and_rejects_file_sources():
    service, threads, _signals = _service()
    service.start()

    assert service.start_recording()
    assert service.is_recording()
    assert service.stop_recording()
    assert not service.is_recording()
    assert service.start_recording()

    service.set_input_file("recording.raw", restart_if_running=True)
    file_thread = threads[1]

    assert threads[0].recording_stop_count == 2
    assert not service.start_recording()
    assert not service.is_recording()
    assert file_thread.recording_start_count == 0


def test_camera_service_reports_failed_recording_start():
    service, threads, _signals = _service()
    service.start()
    threads[0].recording_start_result = False

    assert not service.start_recording()
    assert not service.is_recording()


def test_camera_service_rejects_stale_progress_from_previous_thread():
    service, threads, signals = _service()
    service.set_input_file("first.raw")
    service.start()
    stale_progress = service._thread_progress_handler

    service.set_input_file("second.raw", restart_if_running=True)
    progress_after_switch = list(signals[3].emissions)
    stale_progress(8000, 10000)

    assert signals[3].emissions == progress_after_switch
    assert service._last_progress_current_us == 0
    threads[1].progress_signal.emit(200, 1000)
    assert signals[3].emissions[-1] == (200, 1000)


def test_camera_service_rejects_queued_image_and_status_from_previous_thread():
    service, _threads, signals = _service()
    service.set_input_file("first.raw")
    service.start()
    stale_image = service._thread_image_handler
    stale_status = service._thread_status_handler

    service.set_input_file("second.raw", restart_if_running=True)
    stale_image("old-frame", 123)
    stale_status("old-status")

    assert signals[0].emissions == []
    assert signals[1].emissions == []


def test_camera_service_rejects_queued_finished_from_previous_thread():
    service, _threads, signals = _service()
    service.set_input_file("first.raw")
    service.start()
    stale_finished = service._thread_finished_handler

    service.set_input_file("second.raw", restart_if_running=True)
    finished_after_switch = list(signals[2].emissions)
    stale_finished()

    assert signals[2].emissions == finished_after_switch


def test_camera_service_forwards_finished_for_current_thread():
    service, threads, signals = _service()
    service.set_input_file("events.raw")
    service.start()

    threads[0].finished_signal.emit()

    assert signals[2].emissions == [()]


def test_old_duration_callback_for_same_file_uses_current_seek_fraction():
    metadata = FakeMetadataService()
    service, threads, signals = _service(metadata)
    service.set_input_file("events.raw")
    service.start()
    old_duration_callback = metadata.callback
    threads[0].progress_signal.emit(8000, 10000)

    service.seek(0.25)
    old_duration_callback("events.raw", 20000)

    assert service.last_seek_fraction == 0.25
    assert service._last_progress_current_us == 5000
    assert service._last_progress_total_us == 20000
    assert signals[3].emissions[-1] == (5000, 20000)


def test_camera_service_ignores_stale_source_ready_signal():
    ready_sizes = []
    service, threads, _signals = _service(
        source_ready_callback=lambda width, height: ready_sizes.append((width, height))
    )
    service.start()
    stale_ready = service._thread_source_ready_handler

    service.restart()
    stale_ready(111, 222)

    assert service.current_size() is None
    assert ready_sizes == []
    threads[1].width = 320
    threads[1].height = 240
    threads[1].source_ready_signal.emit(320, 240)
    assert service.current_size() == (320, 240)
    assert ready_sizes == [(320, 240)]


def test_failed_forced_stop_retains_thread_and_aborts_source_switch():
    frame_queue = queue.Queue()
    service, threads, _signals = _service(frame_queue=frame_queue)
    service.start()
    old_thread = threads[0]
    old_thread.wait_results = [False, False, False]
    frame_queue.put_nowait({"msg_type": "EVENTS"})

    with pytest.raises(RuntimeError, match="could not be stopped"):
        service.set_input_file("new.raw", restart_if_running=True)

    assert service.thread is old_thread
    assert service.file_path is None
    assert service.source_mode == SOURCE_MODE_LIVE
    assert not old_thread.deleted
    assert old_thread.interruption_requested
    assert old_thread.terminated
    assert len(threads) == 1
    assert not frame_queue.empty()
