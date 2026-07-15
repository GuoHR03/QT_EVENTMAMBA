from backend.camera_service import CameraService


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
        self.is_recording = False
        self.width = 640
        self.height = 480
        self.started = False
        self.stopped = False
        self.deleted = False

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started and not self.stopped

    def stop(self):
        self.stopped = True

    def wait(self, _timeout=None):
        return True

    def deleteLater(self):
        self.deleted = True


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


def _service(metadata_service=None):
    created_threads = []

    def thread_factory(**kwargs):
        thread = FakeCameraThread(**kwargs)
        created_threads.append(thread)
        return thread

    signals = [FakeSignal() for _ in range(4)]
    service = CameraService(
        frame_queue="frames",
        image_signal=signals[0],
        status_signal=signals[1],
        finished_signal=signals[2],
        progress_signal=signals[3],
        thread_factory=thread_factory,
        metadata_service=metadata_service or FakeMetadataService(),
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
    assert service.current_size() == (640, 480)

    thread.progress_signal.emit(1200, 9000)
    assert metadata.recorded == [("events.raw", 9000)]
    assert signals[3].emissions[-1] == (1200, 9000)

    service.stop()
    assert thread.stopped
    assert thread.deleted
    assert service.thread is None


def test_camera_service_ignores_old_file_duration_callback():
    metadata = FakeMetadataService()
    service, _threads, signals = _service(metadata)
    service.set_input_file("current.raw")
    service._last_progress_current_us = 7000

    service._handle_duration_resolved("old.raw", 10000)
    assert signals[3].emissions == []

    service._handle_duration_resolved("current.raw", 5000)
    assert signals[3].emissions == [(5000, 5000)]
