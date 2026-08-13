import importlib
import os
import threading
from pathlib import Path

import pytest

from backend.inference_service import (
    STATE_ERROR,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    InferenceService,
    _finite_positive_timeout,
)
from backend.backend_healthcheck import StartupCancelledError
from backend.settings import (
    DEFAULT_CENTER_ONNX_MODEL,
    DEFAULT_ELLIPSE_MATRIX,
    DEFAULT_ELLIPSE_ONNX_MODEL,
    DEFAULT_SELECTIVE_SCAN_DLL,
    DEFAULT_WINDOWS_BACKEND_EXECUTABLE,
    DEFAULT_WINDOWS_PYTHON,
)


def _touch(root, relative_path):
    path = root / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _windows_service(root, frozen):
    service = InferenceService.__new__(InferenceService)
    service.windows_python = DEFAULT_WINDOWS_PYTHON
    service.windows_backend_executable = DEFAULT_WINDOWS_BACKEND_EXECUTABLE
    service.center_onnx_model = DEFAULT_CENTER_ONNX_MODEL
    service.ellipse_onnx_model = DEFAULT_ELLIPSE_ONNX_MODEL
    service.ellipse_matrix = DEFAULT_ELLIPSE_MATRIX
    service.selective_scan_dll = DEFAULT_SELECTIVE_SCAN_DLL
    service.active_model_path = None
    service._is_frozen_ui = lambda: frozen

    _touch(root, DEFAULT_CENTER_ONNX_MODEL)
    _touch(root, DEFAULT_ELLIPSE_ONNX_MODEL)
    _touch(root, DEFAULT_ELLIPSE_MATRIX)
    _touch(root, DEFAULT_SELECTIVE_SCAN_DLL)
    return service


def _initialize_lifecycle(service, backend_process=None):
    service.frame_queue = object()
    service.prediction_signal = type(
        "PredictionSignal",
        (),
        {"emit": lambda self, value: None},
    )()
    service.network_thread = None
    service.backend_process = backend_process
    service._state = STATE_STOPPED
    service._state_callback = None
    service.last_error = None
    service._instance_nonce = None
    service._startup_cancelled = threading.Event()
    service._cancel_latched = False
    service._state_lock = threading.RLock()
    service._backend_operation_lock = threading.Lock()
    service._backend_operation = None
    service._backend_ready = False
    service.weights_path = None
    service.prediction_mode = "center"
    return service


def test_source_ui_keeps_python_and_windows_backend_script(tmp_path):
    service = _windows_service(tmp_path, frozen=False)
    python_executable = _touch(tmp_path, DEFAULT_WINDOWS_PYTHON)
    backend_script = _touch(tmp_path, "windows_backend.py")

    command = service._build_windows_command(
        str(tmp_path),
        str(tmp_path / "checkpoint" / "center.pth"),
        "center",
        5555,
    )

    assert command[:2] == [
        os.path.abspath(python_executable),
        os.path.abspath(backend_script),
    ]
    assert service.active_model_path == os.path.abspath(
        tmp_path / DEFAULT_CENTER_ONNX_MODEL
    )


def test_frozen_ui_uses_installed_backend_executable_and_relative_assets(tmp_path):
    service = _windows_service(tmp_path, frozen=True)
    backend_executable = _touch(tmp_path, DEFAULT_WINDOWS_BACKEND_EXECUTABLE)
    selected_model = _touch(tmp_path, "models/selected.onnx")

    command = service._build_windows_command(
        str(tmp_path),
        "models/selected.onnx",
        "ellipse",
        6000,
    )

    assert command[0] == os.path.abspath(backend_executable)
    assert "windows_backend.py" not in command
    assert DEFAULT_WINDOWS_PYTHON not in command
    assert command[command.index("--ellipse-model") + 1] == os.path.abspath(
        selected_model
    )
    assert service.active_model_path == os.path.abspath(selected_model)


def test_frozen_ui_reports_missing_backend_executable_instead_of_python(tmp_path):
    service = _windows_service(tmp_path, frozen=True)
    _touch(tmp_path, DEFAULT_WINDOWS_PYTHON)
    _touch(tmp_path, "windows_backend.py")

    with pytest.raises(FileNotFoundError) as exc_info:
        service._build_windows_command(
            str(tmp_path),
            str(tmp_path / "checkpoint" / "center.pth"),
            "center",
            5555,
        )

    message = str(exc_info.value)
    assert "Windows backend executable" in message
    assert "Windows Python" not in message


def test_is_running_requires_network_thread_and_backend_process():
    transitions = []

    class FakeThread:
        def isRunning(self):
            return True

    class FakeBackend:
        running = True

        def is_running(self):
            return self.running

    service = InferenceService.__new__(InferenceService)
    service.network_thread = FakeThread()
    service.backend_process = FakeBackend()
    service._state = STATE_RUNNING
    service._state_callback = lambda state, error: transitions.append((state, error))
    service.last_error = None
    service._state_lock = threading.RLock()

    assert service.is_running()

    service.backend_process.running = False
    assert not service.is_running()
    assert service.state == STATE_ERROR
    assert "backend process" in service.last_error
    assert transitions[-1][0] == STATE_ERROR


@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan"), "bad"))
def test_ready_timeout_falls_back_to_finite_positive_default(value):
    assert _finite_positive_timeout(value, 180) == 180.0


def test_restart_waits_for_stop_before_starting_again():
    events = []
    service = InferenceService.__new__(InferenceService)
    service.weights_path = "model.onnx"
    service.prediction_mode = "center"
    service.stop_network = lambda: events.append("network stopped")
    service.restart_backend = lambda *args, **kwargs: events.append(
        ("backend restarted", args, kwargs)
    )
    service.start_network = lambda *args, **kwargs: events.append(
        ("network started", args, kwargs)
    )

    InferenceService.restart(service, prediction_mode="ellipse", port=6000)

    assert events[0] == "network stopped"
    assert events[1] == (
        "backend restarted",
        ("ellipse",),
        {"port": 6000, "host": "127.0.0.1"},
    )
    assert events[2] == (
        "network started",
        (),
        {"port": 6000, "host": "127.0.0.1"},
    )


def test_cancel_start_only_interrupts_starting_state():
    service = _initialize_lifecycle(InferenceService.__new__(InferenceService))
    service._state = STATE_STARTING

    assert service.cancel_start()
    assert service._startup_cancelled.is_set()

    service._state = STATE_STOPPING
    service._backend_operation = "restart"
    service._startup_cancelled.clear()
    assert service.cancel_start()
    assert service._startup_cancelled.is_set()

    service._state = STATE_STOPPED
    service._backend_operation = None
    service._startup_cancelled.clear()
    assert not service.cancel_start()
    assert not service._startup_cancelled.is_set()


def test_forced_cancel_is_not_cleared_before_backend_operation_begins():
    class FakeBackend:
        process = None

        def stop(self):
            self.process = None

    service = _initialize_lifecycle(
        InferenceService.__new__(InferenceService),
        FakeBackend(),
    )
    start_calls = []

    def start_backend_locked(*args):
        service._raise_if_start_cancelled()
        start_calls.append(args)

    service._start_backend_locked = start_backend_locked
    assert service.cancel_start(force=True)

    with pytest.raises(StartupCancelledError):
        service.start_backend("model.onnx", "center")

    assert start_calls == []
    assert service.state == STATE_STOPPED
    assert not service._cancel_latched
    assert not service._startup_cancelled.is_set()


def test_start_backend_does_not_create_network_thread():
    class FakeBackend:
        process = None

    service = _initialize_lifecycle(
        InferenceService.__new__(InferenceService),
        FakeBackend(),
    )
    identity = {"status": "READY", "instance_nonce": "test", "pid": 42}
    service._start_backend_locked = lambda *args: identity

    assert service.start_backend("model.onnx", "center") == identity
    assert service.network_thread is None
    assert service._backend_ready
    assert service.state == STATE_STARTING


def test_network_thread_is_created_and_destroyed_by_network_phase_caller(
    monkeypatch,
):
    caller_id = threading.get_ident()
    lifecycle_threads = []

    class FakeResultSignal:
        def connect(self, callback):
            self.callback = callback

    class FakeNetworkThread:
        def __init__(self, *args, **kwargs):
            lifecycle_threads.append(("create", threading.get_ident()))
            self.result_signal = FakeResultSignal()
            self.running = False

        def start(self):
            lifecycle_threads.append(("start", threading.get_ident()))
            self.running = True

        def invalidate_generation(self):
            lifecycle_threads.append(("pause", threading.get_ident()))

        def stop(self):
            lifecycle_threads.append(("stop", threading.get_ident()))
            self.running = False

        def isRunning(self):
            return self.running

        def wait(self, timeout_ms):
            return not self.running

        def terminate(self):
            self.running = False

        def deleteLater(self):
            lifecycle_threads.append(("delete", threading.get_ident()))

    class FakeBackend:
        process = object()

        def is_running(self):
            return True

    network_module = importlib.import_module("backend.NetworkThread")
    monkeypatch.setattr(network_module, "NetworkThread", FakeNetworkThread)
    service = _initialize_lifecycle(
        InferenceService.__new__(InferenceService),
        FakeBackend(),
    )
    service._backend_ready = True
    service._state = STATE_STARTING

    network = service.start_network(start_paused=True)
    assert network is service.network_thread
    assert service.state == STATE_RUNNING
    service.stop_network()

    assert service.network_thread is None
    assert lifecycle_threads == [
        ("create", caller_id),
        ("pause", caller_id),
        ("start", caller_id),
        ("stop", caller_id),
        ("delete", caller_id),
    ]


def test_network_stop_failure_keeps_live_thread_handle():
    class StuckNetworkThread:
        def stop(self):
            pass

        def isRunning(self):
            return True

        def wait(self, timeout_ms):
            return False

        def terminate(self):
            pass

        def deleteLater(self):
            raise AssertionError("a live thread must not be deleted")

    class FakeBackend:
        process = object()

        def is_running(self):
            return True

    service = _initialize_lifecycle(
        InferenceService.__new__(InferenceService),
        FakeBackend(),
    )
    thread = StuckNetworkThread()
    service.network_thread = thread

    with pytest.raises(RuntimeError, match="network thread did not stop"):
        service.stop_network()

    assert service.network_thread is thread
    assert service.state == STATE_ERROR


def test_restart_cancelled_while_stopping_never_starts_backend():
    service = _initialize_lifecycle(InferenceService.__new__(InferenceService))
    service.weights_path = "model.onnx"
    stop_entered = threading.Event()
    allow_stop = threading.Event()
    start_calls = []
    stop_calls = []
    outcome = []

    def stop_backend_locked():
        stop_calls.append("stop")
        if len(stop_calls) == 1:
            stop_entered.set()
            assert allow_stop.wait(2)

    service._stop_backend_locked = stop_backend_locked
    service._start_backend_locked = lambda *args: start_calls.append(args)

    def restart():
        try:
            service.restart_backend("ellipse")
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=restart)
    worker.start()
    assert stop_entered.wait(2)
    assert service.state == STATE_STOPPING
    assert service.cancel_start()
    allow_stop.set()
    worker.join(2)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], StartupCancelledError)
    assert start_calls == []
    assert stop_calls == ["stop", "stop"]
    assert service.state == STATE_STOPPED
    assert service._backend_operation is None


def test_stop_backend_cancels_and_serializes_with_inflight_start():
    events = []

    class FakeBackend:
        process = None
        running = False

        def is_running(self):
            return self.running

        def stop(self):
            events.append("backend stopped")
            self.process = None
            self.running = False

    backend = FakeBackend()
    service = _initialize_lifecycle(
        InferenceService.__new__(InferenceService),
        backend,
    )
    start_entered = threading.Event()
    outcomes = []

    def start_backend_locked(*args):
        backend.process = object()
        backend.running = True
        events.append("backend starting")
        start_entered.set()
        assert service._startup_cancelled.wait(2)
        service._raise_if_start_cancelled()

    service._start_backend_locked = start_backend_locked

    def start():
        try:
            service.start_backend("model.onnx", "center")
        except Exception as exc:
            outcomes.append(exc)

    starter = threading.Thread(target=start)
    stopper = threading.Thread(target=service.stop_backend)
    starter.start()
    assert start_entered.wait(2)
    stopper.start()
    starter.join(2)
    stopper.join(2)

    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], StartupCancelledError)
    assert events == ["backend starting", "backend stopped", "backend stopped"]
    assert service.state == STATE_STOPPED
    assert service._backend_operation is None
