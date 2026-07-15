from backend.event_source import SourceMetadata
from backend.playback_config import PlaybackConfig, PlaybackConfigController
from backend.playback_coordinator import PlaybackCoordinator


class FakeRenderer:
    def __init__(self):
        self.settings = []

    def set_display_settings(self, palette, fps):
        self.settings.append((palette, fps))


class FakeSource:
    def __init__(self, frame_callback):
        self.renderer = FakeRenderer()
        self.device = object()
        self.seek_time_us = 300
        self.frame_callback = frame_callback
        self.close_count = 0
        self.request_stop_count = 0
        self.run_count = 0

    def metadata(self):
        return SourceMetadata(
            source_type="metavision",
            width=320,
            height=240,
            start_time_us=100,
            end_time_us=1100,
            seekable=True,
        )

    def run(self, context, event_pipeline):
        self.run_count += 1
        self.frame_callback(550, [1, 2, 3])
        context.progress_callback(600)

    def request_stop(self):
        self.request_stop_count += 1

    def close(self):
        self.close_count += 1


class FakeNoiseFilter:
    def __init__(self, filter_type, threshold_us, **kwargs):
        self.created_with = (filter_type, threshold_us, kwargs)
        self.initialized = []
        self.updates = []
        self.reset_count = 0

    def initialize(self, width, height):
        self.initialized.append((width, height))

    def update_settings(self, filter_type, threshold_us):
        self.updates.append((filter_type, threshold_us))

    def reset(self):
        self.reset_count += 1

    def apply(self, events):
        return events


class FakeWorker:
    def __init__(self):
        self.start_count = 0
        self.stop_modes = []
        self.wait_count = 0

    def start(self):
        self.start_count += 1

    def stop(self, discard_pending=True):
        self.stop_modes.append(discard_pending)

    def wait(self):
        self.wait_count += 1


class FakeRecorder:
    def __init__(self):
        self.is_recording = False
        self.started_devices = []
        self.stopped_devices = []

    def start(self, device):
        self.started_devices.append(device)
        self.is_recording = device is not None
        return self.is_recording

    def stop(self, device):
        self.stopped_devices.append(device)
        self.is_recording = False
        return device is not None


def test_coordinator_assembles_playback_and_reports_frames_and_progress():
    config_controller = PlaybackConfigController(
        PlaybackConfig(roi=(10, 20, 100, 80), nn_interval_ms=20)
    )
    source_calls = []
    worker_calls = []
    frames = []
    progress = []
    worker = FakeWorker()
    recorder = FakeRecorder()

    def source_factory(**kwargs):
        source_calls.append(kwargs)
        return FakeSource(kwargs["frame_callback"])

    def worker_factory(*args, **kwargs):
        worker_calls.append((args, kwargs))
        return worker

    coordinator = PlaybackCoordinator(
        config_controller=config_controller,
        target_queue="target",
        input_path="events.raw",
        seek_fraction=0.25,
        duration_hint_us=1000,
        frame_callback=lambda frame, timestamp: frames.append((frame, timestamp)),
        progress_callback=lambda current, total: progress.append((current, total)),
        source_factory=source_factory,
        inference_worker_factory=worker_factory,
        noise_filter_factory=FakeNoiseFilter,
        recorder=recorder,
    )

    coordinator.run()

    assert len(source_calls) == 1
    assert source_calls[0]["input_path"] == "events.raw"
    assert source_calls[0]["seek_fraction"] == 0.25
    assert source_calls[0]["duration_hint_us"] == 1000
    assert source_calls[0]["hardware_roi"] == (10, 20, 100, 80)
    assert frames == [([1, 2, 3], 550)]
    assert progress == [(200, 1000), (500, 1000)]
    assert coordinator.width == 320
    assert coordinator.height == 240
    assert coordinator.source_type == "metavision"
    assert coordinator.noise_filter.initialized == [(320, 240)]
    assert coordinator.source.close_count == 1
    assert not coordinator.is_running

    worker_args, worker_kwargs = worker_calls[0]
    assert worker_args[1:4] == (320, 240, "target")
    assert callable(worker_args[4])
    assert worker_kwargs["roi_getter"]() == (10, 20, 100, 80)
    assert worker.start_count == 1
    assert worker.stop_modes == [False]
    assert worker.wait_count == 1

    assert coordinator.start_recording()
    assert coordinator.is_recording
    assert recorder.started_devices == [coordinator.device]
    assert coordinator.stop_recording()
    assert not coordinator.is_recording
    assert recorder.stopped_devices == [coordinator.device]


def test_coordinator_applies_runtime_display_noise_and_roi_updates():
    controller = PlaybackConfigController(PlaybackConfig())
    source = None

    def source_factory(**kwargs):
        nonlocal source
        source = FakeSource(kwargs["frame_callback"])
        return source

    coordinator = PlaybackCoordinator(
        config_controller=controller,
        source_factory=source_factory,
        inference_worker_factory=lambda *args, **kwargs: FakeWorker(),
        noise_filter_factory=FakeNoiseFilter,
    )
    coordinator.run()

    updated = PlaybackConfig(
        palette="Light",
        fps=60,
        replay_factor=2.0,
        roi=(5, 6, 70, 80),
        noise_filter_type="trail",
        noise_filter_threshold_us=2500,
    )
    previous = coordinator.update_config(updated)

    assert previous == PlaybackConfig()
    assert controller.get() == updated
    assert source.renderer.settings == [("Light", 60.0)]
    assert coordinator.noise_filter.updates == [("trail", 2500)]
    assert coordinator.roi_tuple() == (5, 6, 70, 80)

    coordinator.update_config(updated.with_updates(roi=None))
    assert coordinator.noise_filter.reset_count == 1


def test_coordinator_stopped_before_run_does_not_open_source():
    source_calls = []
    coordinator = PlaybackCoordinator(
        source_factory=lambda **kwargs: source_calls.append(kwargs),
        inference_worker_factory=lambda *args, **kwargs: FakeWorker(),
        noise_filter_factory=FakeNoiseFilter,
    )

    coordinator.stop()
    coordinator.run()

    assert source_calls == []
    assert not coordinator.is_running
