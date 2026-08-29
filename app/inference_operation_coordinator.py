"""Coordinate asynchronous inference operations outside the main window."""

from backend.inference_lifecycle import (
    STATE_ERROR,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPING,
)

from .inference_operation import InferenceOperationThread
from .inference_operation_state import (
    INFERENCE_CLEANUP,
    INFERENCE_CLOSE,
    INFERENCE_RESTART,
    INFERENCE_START,
    INFERENCE_STOP,
    InferenceOperationState,
    inference_operation_action,
)


class InferenceOperationCoordinator:
    """Own worker callbacks and recovery ordering for inference operations."""

    def __init__(self, window, state=None, worker_factory=None, defer=None):
        self.window = window
        self.controller = window.controller
        self.view_state = window.view_state
        self.predictions = window.predictions
        self.state = state or InferenceOperationState()
        self.worker_factory = worker_factory or InferenceOperationThread
        self.defer = defer or _defer_to_qt_event_loop

    @property
    def busy(self):
        # Busy lasts until queued result and finished signals are both handled.
        return self.state.busy

    def stop_network_before_backend(self, action):
        try:
            self.controller.stop_model_network()
        except Exception as exc:
            self.view_state.set_model_error()
            self.window.append_log(
                f"{action}推理前停止网络线程失败：{exc}",
                "error",
            )
            self._observe_runtime_state()
            return False
        return True

    def start(self, operation_name, operation, allow_when_closing=False):
        if self.state.close_pending and not allow_when_closing:
            return False
        if self.busy:
            self.window.append_log("已有推理服务操作正在进行，请稍候", "warning")
            return False

        worker = self.worker_factory(operation_name, operation, self.window)
        worker.succeeded.connect(self.handle_success)
        worker.failed.connect(self.handle_failure)
        worker.cancelled.connect(self.handle_cancelled)
        worker.finished.connect(self.finish)
        if not self.state.attach(worker):
            worker.deleteLater()
            return False

        self.window._set_prediction_mode_controls_enabled(False)
        try:
            worker.start()
        except Exception as exc:
            self.state.detach()
            worker.deleteLater()
            self.window._set_prediction_mode_controls_enabled(True)
            self.handle_failure(operation_name, str(exc))
            return False
        return True

    def handle_success(self, operation_name):
        if self.state.close_pending or operation_name == INFERENCE_CLOSE:
            return

        runtime_name = self.controller.inference_runtime_display_name
        if operation_name == INFERENCE_STOP:
            self.view_state.set_model_stopped()
            self.predictions.clear()
            self.window.append_log(f"{runtime_name} 推理服务已停止", "info")
        elif operation_name == INFERENCE_CLEANUP:
            self.view_state.set_model_stopped()
            self.predictions.clear()
            self.window.append_log("失败操作残留的推理后端已清理", "info")
        elif operation_name in (INFERENCE_START, INFERENCE_RESTART):
            try:
                self.controller.start_model_network()
            except Exception as exc:
                self.handle_network_start_failure(operation_name, exc)
                return
            if self.controller.active_model_path:
                self.view_state.set_weight_file(self.controller.active_model_path)
            self.view_state.set_model_running()
            action = "启动" if operation_name == INFERENCE_START else "重启"
            self.window.append_log(
                f"{runtime_name} 推理服务已{action}",
                "success",
            )
        self._observe_runtime_state()

    def handle_network_start_failure(self, operation_name, error):
        cleanup_details = ""
        try:
            self.controller.stop_model_network()
        except Exception as cleanup_error:
            cleanup_details = f"；网络线程清理失败：{cleanup_error}"

        self.state.request_cleanup()
        self.view_state.set_model_error()
        action = "启动" if operation_name == INFERENCE_START else "重启"
        self.window.append_log(
            f"{action}推理网络失败：{error}{cleanup_details}；正在清理后端",
            "error",
        )
        self._observe_runtime_state()

    def handle_failure(self, operation_name, message):
        action = inference_operation_action(operation_name)
        self.window.append_log(f"{action}失败：{message}", "error")

        if operation_name == INFERENCE_CLOSE:
            self.state.abort_close()
            self.window.setEnabled(True)
            self.window._inference_health_timer.start()
            self.view_state.set_model_error()
            self._observe_runtime_state()
            return
        if self.state.close_pending:
            return
        self.view_state.set_model_error()
        self._observe_runtime_state()

    def handle_cancelled(self, operation_name):
        if operation_name == INFERENCE_CLOSE:
            self.state.abort_close()
            self.window.setEnabled(True)
            self.window._inference_health_timer.start()
            self.view_state.set_model_error()
            self.window.append_log("关闭清理被取消，窗口保持打开", "warning")
            self._observe_runtime_state()
            return
        if self.state.close_pending:
            return
        self.view_state.set_model_error()
        self.window.append_log(f"推理操作已取消：{operation_name}", "warning")
        self._observe_runtime_state()

    def finish(self):
        worker = self.state.detach()
        operation_name = None
        if worker is not None:
            operation_name = worker.operation_name
            worker.deleteLater()

        if self.state.close_pending:
            self.state.clear_cleanup()
            if operation_name == INFERENCE_CLOSE:
                self.window._complete_close()
            else:
                self.defer(self.window._begin_close_cleanup)
            return

        if self.state.take_cleanup():
            self.view_state.set_model_stopping()
            self.defer(self.start_backend_cleanup)
            return

        self.window._set_prediction_mode_controls_enabled(True)

    def start_backend_cleanup(self):
        if self.state.close_pending:
            self.window._begin_close_cleanup()
            return
        started = self.start(
            INFERENCE_CLEANUP,
            self.controller.unload_model,
        )
        if not started:
            self.window._set_prediction_mode_controls_enabled(True)
            self.view_state.set_model_error()
            self.window.append_log(
                "无法启动推理后端清理任务",
                "error",
            )

    def refresh_runtime_state(self):
        if self.state.close_pending or self.busy:
            return
        if self.controller.inference_state == STATE_RUNNING:
            self.controller.is_inference_running()
        state = self.controller.inference_state
        if not self.state.observe_runtime_state(state):
            return

        if state == STATE_RUNNING:
            self.view_state.set_model_running()
        elif state == STATE_STARTING:
            self.view_state.set_model_starting()
        elif state == STATE_STOPPING:
            self.view_state.set_model_stopping()
        elif state == STATE_ERROR:
            self.view_state.set_model_error()
            if self.controller.inference_last_error:
                self.window.append_log(
                    f"推理服务异常：{self.controller.inference_last_error}",
                    "error",
                )
        else:
            self.view_state.set_model_stopped()

    def _observe_runtime_state(self):
        self.state.observe_runtime_state(self.controller.inference_state)


def _defer_to_qt_event_loop(callback):
    # Keep module import and unit testing independent from a concrete Qt
    # runtime; the real UI resolves QTimer only when deferred work is needed.
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(0, callback)
