import threading

import pytest

from app.inference_operation_state import InferenceOperationState
from app.shutdown_coordinator import (
    SHUTDOWN_BACKEND,
    SHUTDOWN_COMPLETE,
    SHUTDOWN_FAILED,
    SHUTDOWN_FINALIZING,
    SHUTDOWN_IDLE,
    SHUTDOWN_UI_RESOURCES,
    ApplicationShutdownCoordinator,
)


class FakeController:
    def __init__(self):
        self.calls = []
        self.ui_failures = []
        self.backend_failures = []

    def close_ui_resources(self):
        self.calls.append("ui")
        if self.ui_failures:
            raise self.ui_failures.pop(0)
        return True

    def close_backend_resources(self):
        self.calls.append("backend")
        if self.backend_failures:
            raise self.backend_failures.pop(0)
        return True


def _coordinator(controller=None):
    controller = controller or FakeController()
    state = InferenceOperationState()
    return ApplicationShutdownCoordinator(controller, state), controller, state


def test_shutdown_coordinator_enforces_order_and_commits_once():
    shutdown, controller, state = _coordinator()

    assert shutdown.phase == SHUTDOWN_IDLE
    assert shutdown.begin()
    assert not shutdown.begin()
    assert shutdown.phase == SHUTDOWN_UI_RESOURCES

    shutdown.close_ui_resources()
    assert shutdown.phase == SHUTDOWN_BACKEND
    shutdown.close_ui_resources()
    shutdown.close_backend_resources()
    assert shutdown.phase == SHUTDOWN_FINALIZING
    shutdown.close_backend_resources()
    shutdown.complete()

    assert shutdown.phase == SHUTDOWN_COMPLETE
    assert shutdown.ready
    assert state.close_ready
    assert controller.calls == ["ui", "backend"]


def test_concurrent_close_requests_are_coalesced():
    shutdown, _controller, _state = _coordinator()
    barrier = threading.Barrier(8)
    results = []

    def begin_close():
        barrier.wait()
        results.append(shutdown.begin())

    threads = [threading.Thread(target=begin_close) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1)

    assert sum(results) == 1
    assert shutdown.pending


def test_shutdown_coordinator_rejects_backend_before_ui_phase():
    shutdown, controller, _state = _coordinator()
    shutdown.begin()

    with pytest.raises(RuntimeError, match="UI resources must close"):
        shutdown.close_backend_resources()

    assert controller.calls == []
    shutdown.abort("out of order")
    assert shutdown.phase == SHUTDOWN_FAILED


def test_backend_failure_retry_resumes_without_reclosing_ui_resources():
    controller = FakeController()
    controller.backend_failures.append(RuntimeError("backend busy"))
    shutdown, controller, _state = _coordinator(controller)

    shutdown.begin()
    shutdown.close_ui_resources()
    with pytest.raises(RuntimeError, match="backend busy"):
        shutdown.close_backend_resources()

    assert shutdown.phase == SHUTDOWN_FAILED
    assert shutdown.last_error == "backend busy"
    shutdown.abort()

    assert shutdown.begin()
    shutdown.close_ui_resources()
    shutdown.close_backend_resources()
    shutdown.complete()

    assert controller.calls == ["ui", "backend", "backend"]
    assert shutdown.errors == ((SHUTDOWN_BACKEND, "backend busy"),)


def test_ui_failure_retries_the_ui_phase_before_backend():
    controller = FakeController()
    controller.ui_failures.append(RuntimeError("camera timeout"))
    shutdown, controller, _state = _coordinator(controller)

    shutdown.begin()
    with pytest.raises(RuntimeError, match="camera timeout"):
        shutdown.close_ui_resources()
    shutdown.abort()

    assert shutdown.begin()
    shutdown.close_ui_resources()
    shutdown.close_backend_resources()
    shutdown.complete()

    assert controller.calls == ["ui", "ui", "backend"]
    assert shutdown.errors == ((SHUTDOWN_UI_RESOURCES, "camera timeout"),)
