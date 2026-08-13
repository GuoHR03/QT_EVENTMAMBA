from app.inference_operation import InferenceOperationThread


class ControllableOperationThread(InferenceOperationThread):
    """Make cancellation deterministic when run() is invoked synchronously."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_cancelled = False

    def requestInterruption(self):
        self._test_cancelled = True

    def isInterruptionRequested(self):
        return self._test_cancelled


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
