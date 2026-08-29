from app.inference_operation import InferenceOperationThread
from app.inference_operation_state import (
    INFERENCE_CLOSE,
    InferenceOperationState,
    inference_operation_action,
)


class ControllableOperationThread(InferenceOperationThread):
    """Make cancellation deterministic when run() is invoked synchronously."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_cancelled = False

    def requestInterruption(self):
        self._test_cancelled = True

    def isInterruptionRequested(self):
        return self._test_cancelled


def test_operation_state_owns_busy_close_cleanup_and_runtime_transitions():
    state = InferenceOperationState()
    worker = type("Worker", (), {"operation_name": INFERENCE_CLOSE})()

    assert state.attach(worker)
    assert state.busy
    assert state.operation_name == INFERENCE_CLOSE
    assert not state.attach(object())
    assert state.begin_close()
    assert not state.begin_close()

    state.request_cleanup()
    assert state.take_cleanup()
    assert not state.take_cleanup()
    assert state.observe_runtime_state("running")
    assert not state.observe_runtime_state("running")
    assert state.detach() is worker
    assert not state.busy


def test_operation_action_has_a_stable_fallback():
    assert inference_operation_action("start") == "启动推理"
    assert inference_operation_action("custom") == "推理操作"


def test_inference_operation_emits_success_after_operation():
    calls = []
    worker = ControllableOperationThread(
        "start",
        lambda: calls.append("operation"),
    )
    worker.succeeded.connect(lambda name: calls.append(("succeeded", name)))
    worker.failed.connect(lambda name, message: calls.append(("failed", name, message)))
    worker.cancelled.connect(lambda name: calls.append(("cancelled", name)))

    worker.run()

    assert calls == ["operation", ("succeeded", "start")]


def test_inference_operation_does_not_run_when_already_cancelled():
    calls = []
    worker = ControllableOperationThread(
        "start",
        lambda: calls.append("operation"),
    )
    worker.cancelled.connect(lambda name: calls.append(("cancelled", name)))
    worker.requestInterruption()

    worker.run()

    assert calls == [("cancelled", "start")]


def test_inference_operation_rechecks_cancellation_after_blocking_call():
    calls = []
    worker = None

    def operation():
        calls.append("operation")
        worker.requestInterruption()

    worker = ControllableOperationThread("restart", operation)
    worker.succeeded.connect(lambda name: calls.append(("succeeded", name)))
    worker.cancelled.connect(lambda name: calls.append(("cancelled", name)))

    worker.run()

    assert calls == ["operation", ("cancelled", "restart")]
