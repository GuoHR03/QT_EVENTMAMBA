from pathlib import Path
from types import SimpleNamespace

from app.controller import AppController


class FakeBackend:
    def __init__(self):
        self.running = False
        self.recording = False
        self.source_mode = "live"
        self.input_calls = []
        self.live_calls = []
        self.recording_start_result = True
        self.recording_stop_result = True
        self.inference = SimpleNamespace(state="stopped", last_error=None)
        self.restart_calls = 0
        self.cancel_start_calls = 0
        self.lifecycle_calls = []

    def is_camera_running(self):
        return self.running

    def is_recording(self):
        return self.recording

    def set_input_file(self, file_path, config=None, restart_if_running=False):
        self.input_calls.append((file_path, config, restart_if_running))
        self.source_mode = "file"
        return self.running and restart_if_running

    def set_live_camera(self, config=None):
        self.live_calls.append(config)
        self.source_mode = "live"
        return self.running

    def start_recording(self):
        if self.recording_start_result:
            self.recording = True
        return self.recording_start_result

    def stop_recording(self):
        if self.recording_stop_result:
            self.recording = False
        return self.recording_stop_result

    def stop_camera(self):
        self.lifecycle_calls.append(("stop_camera",))
        self.running = False

    def start_eventmamba_backend(self, weights_path):
        self.lifecycle_calls.append(("start_backend", weights_path))

    def start_eventmamba_network(self):
        self.lifecycle_calls.append(("start_network",))
        return "network-thread"

    def stop_eventmamba_network(self):
        self.lifecycle_calls.append(("stop_network",))

    def stop_eventmamba_backend(self):
        self.lifecycle_calls.append(("stop_backend",))

    def restart_eventmamba_backend(self):
        self.lifecycle_calls.append(("restart_backend",))
        self.restart_calls += 1

    def cancel_eventmamba_start(self):
        self.cancel_start_calls += 1
        return True


def _controller(backend=None):
    controller = AppController.__new__(AppController)
    controller.settings = SimpleNamespace(playback_config="config")
    controller.backend = backend or FakeBackend()
    controller.input_file_path = None
    controller.weights_path = None
    return controller


def test_set_input_file_accepts_raw_and_propagates_restart_state():
    backend = FakeBackend()
    backend.running = True
    controller = _controller(backend)
    raw_path = Path("recordings") / "events.raw"

    restarted = controller.set_input_file(raw_path, restart_if_running=True)

    assert restarted
    assert controller.input_file_path == str(raw_path)
    assert controller.source_mode == "file"
    assert backend.input_calls == [(str(raw_path), "config", True)]


def test_set_live_camera_clears_file_path_and_preserves_running_state():
    backend = FakeBackend()
    backend.running = True
    backend.source_mode = "file"
    controller = _controller(backend)
    controller.input_file_path = "events.raw"

    restarted = controller.set_live_camera()

    assert restarted
    assert controller.input_file_path is None
    assert controller.source_mode == "live"
    assert backend.live_calls == ["config"]


def test_toggle_recording_uses_backend_result_and_does_not_fake_file_recording():
    backend = FakeBackend()
    backend.running = True
    backend.source_mode = "file"
    backend.recording_start_result = False
    controller = _controller(backend)
    controller.input_file_path = "events.raw"

    assert not controller.start_recording()
    assert controller.toggle_recording() is False
    assert not controller.is_recording()

    backend.source_mode = "live"
    backend.recording_start_result = True
    controller.input_file_path = None
    assert controller.toggle_recording() is True
    assert controller.is_recording()
    assert controller.toggle_recording() is False
    assert not controller.is_recording()


def test_toggle_recording_keeps_state_when_stop_fails():
    backend = FakeBackend()
    backend.running = True
    backend.recording = True
    backend.recording_stop_result = False
    controller = _controller(backend)

    assert controller.toggle_recording() is None
    assert controller.is_recording()


def test_controller_exposes_inference_restart_cancel_and_state():
    backend = FakeBackend()
    backend.inference.state = "error"
    backend.inference.last_error = "backend exited"
    controller = _controller(backend)
    controller.weights_path = "model.onnx"

    controller.restart_model()

    assert backend.restart_calls == 1
    assert controller.cancel_model_start()
    assert backend.cancel_start_calls == 1
    assert controller.inference_state == "error"
    assert controller.inference_last_error == "backend exited"


def test_controller_exposes_split_inference_lifecycle_phases():
    backend = FakeBackend()
    controller = _controller(backend)
    controller.set_weights_path("model.onnx")

    controller.load_model()
    assert controller.start_model_network() == "network-thread"
    controller.stop_model_network()
    controller.restart_model()
    controller.unload_model()

    assert backend.lifecycle_calls == [
        ("start_backend", "model.onnx"),
        ("start_network",),
        ("stop_network",),
        ("restart_backend",),
        ("stop_backend",),
    ]


def test_controller_splits_ui_and_backend_close_resources():
    backend = FakeBackend()
    backend.running = True
    controller = _controller(backend)

    controller.close_ui_resources()
    controller.close_backend_resources()

    assert backend.lifecycle_calls == [
        ("stop_camera",),
        ("stop_network",),
        ("stop_backend",),
    ]
