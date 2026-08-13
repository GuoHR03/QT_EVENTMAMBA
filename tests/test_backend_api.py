import queue

import pytest

from backend.api import BackendAPI
from backend.playback_config import PlaybackConfig


class FakeNetworkThread:
    def __init__(self):
        self.running = True
        self.invalidate_count = 0
        self.resume_count = 0
        self.pending_payload = None
        self.generation = object()

    def isRunning(self):
        return self.running

    def invalidate_generation(self):
        self.invalidate_count += 1
        self.generation = object()
        return self.generation

    def resume_generation(self):
        self.resume_count += 1

    def replace_pending_payload(self, payload):
        self.pending_payload = payload

    def is_generation_current(self, generation):
        return generation is self.generation


class FakeInference:
    def __init__(self, network_thread):
        self.network_thread = network_thread
        self.start_calls = []
        self.restart_calls = []

    def start_backend(self, weights_path, mode, port=5555, host="127.0.0.1"):
        self.start_calls.append((weights_path, mode, port, host))
        return "identity"

    def start_network(self, host="127.0.0.1", port=5555, start_paused=False):
        assert start_paused
        self.network_thread.invalidate_generation()
        return self.network_thread

    def restart_backend(self, mode, port=5555, host="127.0.0.1"):
        self.restart_calls.append((mode, port, host))

    def stop_network(self):
        self.network_thread.running = False
        return True

    def stop_backend(self):
        return True

    def stop(self):
        self.network_thread.running = False
        return True


class FakeCamera:
    def __init__(self, size=None):
        self.running = True
        self.size = size
        self.start_calls = []
        self.restart_calls = []
        self.stop_error = None
        self.apply_calls = []
        self.analysis_enabled_calls = []

    def is_running(self):
        return self.running

    def current_size(self):
        return self.size

    def start(self, config=None):
        self.start_calls.append(config)

    def restart(self, config=None):
        self.restart_calls.append(config)

    def apply_config(self, config):
        self.apply_calls.append(config)
        return False

    def stop(self, emit_finished=True):
        if self.stop_error is not None:
            raise self.stop_error
        self.running = False

    def set_analysis_enabled(self, enabled):
        self.analysis_enabled_calls.append(bool(enabled))
        return True


def _api(camera_size=None):
    api = BackendAPI()
    network_thread = FakeNetworkThread()
    api.camera_queue = queue.Queue(maxsize=1)
    api.camera = FakeCamera(size=camera_size)
    api.inference = FakeInference(network_thread)
    api._pending_camera_network_thread = None
    return api, api.camera, api.inference, network_thread


def test_camera_start_pauses_network_until_real_source_size_is_ready():
    api, camera, _inference, network_thread = _api()

    api.start_camera(config="playback-config")

    assert camera.start_calls == ["playback-config"]
    assert network_thread.invalidate_count == 1
    assert network_thread.resume_count == 0
    assert api._pending_camera_network_thread is network_thread

    api._handle_camera_source_ready(320, 240)

    assert network_thread.pending_payload == {
        "msg_type": "CONFIG",
        "width": 320,
        "height": 240,
        "prediction_mode": "center",
    }
    assert network_thread.resume_count == 1
    assert api._pending_camera_network_thread is None


def test_camera_transition_failure_resumes_network_and_propagates_error():
    api, camera, _inference, network_thread = _api()
    camera.stop_error = RuntimeError("camera did not stop")

    with pytest.raises(RuntimeError, match="did not stop"):
        api.stop_camera()

    assert network_thread.invalidate_count == 1
    assert network_thread.resume_count == 1
    assert api._pending_camera_network_thread is None


def test_inference_start_uses_ready_camera_dimensions_before_resuming():
    api, _camera, inference, network_thread = _api(camera_size=(1280, 720))

    api.start_eventmamba("weights.onnx", port=6000, host="localhost")

    assert inference.start_calls == [
        ("weights.onnx", "center", 6000, "localhost")
    ]
    assert network_thread.invalidate_count == 1
    assert network_thread.pending_payload["width"] == 1280
    assert network_thread.pending_payload["height"] == 720
    assert network_thread.resume_count == 1
    assert _camera.analysis_enabled_calls[-1] is True


def test_inference_start_waits_when_camera_metadata_is_not_ready():
    api, _camera, _inference, network_thread = _api(camera_size=None)

    api.start_eventmamba("weights.onnx")

    assert network_thread.invalidate_count == 1
    assert network_thread.resume_count == 0
    assert network_thread.pending_payload is None
    assert _camera.analysis_enabled_calls[-1] is False

    api._handle_camera_source_ready(346, 260)
    assert network_thread.pending_payload["width"] == 346
    assert network_thread.pending_payload["height"] == 260
    assert network_thread.resume_count == 1
    assert _camera.analysis_enabled_calls[-1] is True


def test_stopping_inference_network_disables_camera_generation_gate():
    api, camera, _inference, _network_thread = _api(camera_size=(640, 480))
    camera.set_analysis_enabled(True)

    assert api.stop_eventmamba_network()

    assert camera.analysis_enabled_calls[-1] is False


def test_failed_network_stop_cannot_be_resumed_by_late_source_ready():
    api, camera, inference, network_thread = _api(camera_size=(640, 480))
    camera.set_analysis_enabled(True)

    def fail_stop():
        inference.state = "error"
        raise RuntimeError("network did not stop")

    inference.stop_network = fail_stop
    with pytest.raises(RuntimeError, match="did not stop"):
        api.stop_eventmamba_network()
    resumes_before_ready = network_thread.resume_count

    api._handle_camera_source_ready(640, 480)

    assert network_thread.resume_count == resumes_before_ready
    assert camera.analysis_enabled_calls[-1] is False


def test_config_fallback_replaces_full_queue_without_empty_race():
    api, _camera, _inference, _network_thread = _api(camera_size=(320, 240))
    api.camera_queue.put_nowait({"msg_type": "EVENTS"})

    assert api._enqueue_camera_config(network_thread=object())

    assert api.camera_queue.get_nowait() == {
        "msg_type": "CONFIG",
        "width": 320,
        "height": 240,
        "prediction_mode": "center",
    }


def test_current_finished_signal_resumes_network_after_source_open_failure():
    api, _camera, _inference, network_thread = _api()
    api._pending_camera_network_thread = network_thread

    api._handle_camera_finished()

    assert network_thread.resume_count == 1
    assert api._pending_camera_network_thread is None


def test_hot_update_reinstalls_config_before_network_resume():
    api, camera, _inference, network_thread = _api(camera_size=(1280, 720))
    network_thread.pending_payload = {"msg_type": "CONFIG", "width": 1}
    config = PlaybackConfig(roi=(800, 100, 200, 200))

    assert not api.update_playback_config(config)

    assert camera.apply_calls == [config]
    assert network_thread.invalidate_count == 1
    assert network_thread.pending_payload == {
        "msg_type": "CONFIG",
        "width": 1280,
        "height": 720,
        "prediction_mode": "center",
    }
    assert network_thread.resume_count == 1


def test_queued_old_generation_prediction_is_rejected_at_ui_boundary():
    api, _camera, _inference, network_thread = _api(camera_size=(640, 480))
    delivered = []
    api.prediction_signal.connect(
        lambda result, timestamp: delivered.append((result, timestamp))
    )
    old_generation = network_thread.generation

    network_thread.invalidate_generation()
    api._handle_network_result(
        {"msg_type": "PREDICTION", "values": [0.5, 0.5]},
        10,
        old_generation,
    )
    api._handle_network_result(
        {"msg_type": "PREDICTION", "values": [0.25, 0.75]},
        20,
        network_thread.generation,
    )

    assert delivered == [
        ({"msg_type": "PREDICTION", "values": [0.25, 0.75]}, 20)
    ]
