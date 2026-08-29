"""Inference operation names and UI state without Qt dependencies."""


INFERENCE_START = "start"
INFERENCE_STOP = "stop"
INFERENCE_RESTART = "restart"
INFERENCE_CLEANUP = "cleanup"
INFERENCE_CLOSE = "close"

_OPERATION_ACTIONS = {
    INFERENCE_START: "启动推理",
    INFERENCE_STOP: "停止推理",
    INFERENCE_RESTART: "重启推理",
    INFERENCE_CLEANUP: "清理推理后端",
    INFERENCE_CLOSE: "关闭推理后端",
}


def inference_operation_action(operation_name):
    return _OPERATION_ACTIONS.get(operation_name, "推理操作")


class InferenceOperationState:
    """Own the UI operation flags that must change atomically together."""

    def __init__(self):
        self.worker = None
        self.close_pending = False
        self.close_ready = False
        self.cleanup_pending = False
        self.last_runtime_state = None

    @property
    def busy(self):
        return self.worker is not None

    @property
    def operation_name(self):
        return getattr(self.worker, "operation_name", None)

    def attach(self, worker):
        if self.busy:
            return False
        self.worker = worker
        return True

    def detach(self):
        worker = self.worker
        self.worker = None
        return worker

    def begin_close(self):
        if self.close_pending:
            return False
        self.close_pending = True
        return True

    def abort_close(self):
        self.close_pending = False

    def complete_close(self):
        self.close_ready = True

    def request_cleanup(self):
        self.cleanup_pending = True

    def clear_cleanup(self):
        self.cleanup_pending = False

    def take_cleanup(self):
        pending = self.cleanup_pending
        self.cleanup_pending = False
        return pending

    def observe_runtime_state(self, state):
        if state == self.last_runtime_state:
            return False
        self.last_runtime_state = state
        return True
