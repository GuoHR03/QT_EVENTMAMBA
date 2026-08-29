from types import SimpleNamespace

from app.inference_operation_coordinator import InferenceOperationCoordinator
from app.inference_operation_state import (
    INFERENCE_CLOSE,
    INFERENCE_START,
)


class FakeSignal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class FakeWorker:
    def __init__(self, operation_name, operation, _parent):
        self.operation_name = operation_name
        self.operation = operation
        self.succeeded = FakeSignal()
        self.failed = FakeSignal()
        self.cancelled = FakeSignal()
        self.finished = FakeSignal()
        self.started = False
        self.deleted = False

    def start(self):
        self.started = True

    def deleteLater(self):
        self.deleted = True


class FakeController:
    def __init__(self):
        self.inference_state = "starting"
        self.inference_last_error = None
        self.inference_runtime_display_name = "Test Runtime"
        self.active_model_path = "model.onnx"
        self.network_starts = 0
        self.network_stops = 0
        self.fail_network_start = False
        self.unload_calls = 0

    def start_model_network(self):
        self.network_starts += 1
        if self.fail_network_start:
            raise RuntimeError("network failed")
        self.inference_state = "running"

    def stop_model_network(self):
        self.network_stops += 1

    def unload_model(self):
        self.unload_calls += 1

    def is_inference_running(self):
        return self.inference_state == "running"


class FakeViewState:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("set_model_") or name == "set_weight_file":
            return lambda *args: self.calls.append((name, args))
        raise AttributeError(name)


class FakeWindow:
    def __init__(self):
        self.controller = FakeController()
        self.view_state = FakeViewState()
        self.predictions = SimpleNamespace(clear=lambda: None)
        self.logs = []
        self.mode_control_states = []
        self.enabled_states = []
        self.close_cleanup_calls = 0
        self.complete_close_calls = 0
        self._inference_health_timer = SimpleNamespace(
            start=lambda: self.logs.append(("timer", "start"))
        )

    def append_log(self, message, level):
        self.logs.append((message, level))

    def _set_prediction_mode_controls_enabled(self, enabled):
        self.mode_control_states.append(bool(enabled))

    def setEnabled(self, enabled):
        self.enabled_states.append(bool(enabled))

    def _begin_close_cleanup(self):
        self.close_cleanup_calls += 1

    def _complete_close(self):
        self.complete_close_calls += 1


def _coordinator(window=None, deferred=None):
    window = window or FakeWindow()
    deferred = [] if deferred is None else deferred
    coordinator = InferenceOperationCoordinator(
        window,
        worker_factory=FakeWorker,
        defer=deferred.append,
    )
    return coordinator, window, deferred


def test_coordinator_starts_worker_and_commits_network_success():
    coordinator, window, _deferred = _coordinator()

    assert coordinator.start(INFERENCE_START, lambda: None)
    worker = coordinator.state.worker
    assert worker.started
    assert window.mode_control_states == [False]

    coordinator.handle_success(INFERENCE_START)
    coordinator.finish()

    assert window.controller.network_starts == 1
    assert ("set_model_running", ()) in window.view_state.calls
    assert worker.deleted
    assert window.mode_control_states[-1] is True


def test_network_start_failure_schedules_backend_cleanup_after_finish():
    coordinator, window, deferred = _coordinator()
    window.controller.fail_network_start = True
    coordinator.start(INFERENCE_START, lambda: None)

    coordinator.handle_success(INFERENCE_START)
    coordinator.finish()

    assert window.controller.network_stops == 1
    assert len(deferred) == 1
    assert ("set_model_error", ()) in window.view_state.calls
    assert ("set_model_stopping", ()) in window.view_state.calls

    deferred.pop()()
    assert coordinator.busy
    assert coordinator.state.operation_name == "cleanup"


def test_close_failure_reopens_window_and_clears_close_state():
    coordinator, window, _deferred = _coordinator()
    coordinator.state.begin_close()

    coordinator.handle_failure(INFERENCE_CLOSE, "stop failed")

    assert not coordinator.state.close_pending
    assert window.enabled_states == [True]
    assert ("timer", "start") in window.logs
