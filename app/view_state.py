import os


def source_is_file(controller):
    """Prefer the controller's explicit source mode, with path compatibility."""
    mode = getattr(controller, "source_mode", None)
    mode = getattr(mode, "value", mode)
    if mode is not None:
        normalized = str(mode).strip().lower()
        if normalized in {"live", "camera", "realtime", "live_camera"}:
            return False
        if normalized in {"raw", "file", "playback", "recording"}:
            return True
    return bool(getattr(controller, "input_file_path", None))


class MainViewState:
    def __init__(self, view):
        self.view = view

    def set_camera_running(self):
        controller = getattr(self.view, "controller", None)
        file_mode = source_is_file(controller)
        self.view.start_camera_button.setText("停止播放" if file_mode else "停止相机")
        self._set_button_role(self.view.start_camera_button, "primary")
        self._set_status("camera", "运行中", "active")
        self.set_recording_stopped(enabled=not file_mode)

    def set_camera_stopped(self):
        controller = getattr(self.view, "controller", None)
        file_mode = source_is_file(controller)
        self.view.start_camera_button.setText("开始播放" if file_mode else "启动相机")
        self._set_button_role(self.view.start_camera_button, "primary")
        self._set_status("camera", "已停止", "idle")

    def set_recording_running(self):
        self.view.record_button.setEnabled(True)
        self.view.record_button.setText("停止录制")
        self._set_button_role(self.view.record_button, "danger")
        self._set_status("camera", "录制中", "danger")

    def set_recording_stopped(self, enabled):
        self.view.record_button.setEnabled(enabled)
        self.view.record_button.setText("录制 RAW")
        self._set_button_role(self.view.record_button, "dangerOutline")
        if enabled:
            self._set_status("camera", "运行中", "active")

    def set_model_starting(self):
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(False)
        self.view.restart_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(False)
        self.view.load_model_button.setText("启动中...")
        self._set_status("model", "推理启动中", "pending")

    def set_model_running(self):
        self.view.load_model_button.setText("推理运行中")
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(True)
        self.view.restart_model_button.setEnabled(True)
        self.view.select_weight_button.setEnabled(False)
        self._set_status("model", "推理运行中", "active")

    def set_model_stopping(self):
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(False)
        self.view.restart_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(False)
        self.view.unload_model_button.setText("停止中...")
        self._set_status("model", "推理停止中", "pending")

    def set_model_stopped(self):
        self.view.load_model_button.setEnabled(True)
        self.view.unload_model_button.setEnabled(False)
        self.view.restart_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(True)
        self.view.load_model_button.setText("启动推理")
        self.view.unload_model_button.setText("停止推理")
        self.view.restart_model_button.setText("重启推理")
        self._set_button_role(self.view.load_model_button, "primary")
        self._set_status("model", "推理已停止", "idle")

    def set_model_error(self):
        self.view.load_model_button.setEnabled(True)
        # Cleanup remains available because a failed stop may deliberately
        # retain a live process handle for a later retry.
        self.view.unload_model_button.setEnabled(True)
        self.view.restart_model_button.setEnabled(True)
        self.view.select_weight_button.setEnabled(False)
        self.view.load_model_button.setText("重试启动")
        self.view.unload_model_button.setText("清理服务")
        self.view.restart_model_button.setText("重试重启")
        self._set_button_role(self.view.load_model_button, "primary")
        self._set_status("model", "推理错误", "danger")

    # Compatibility names for callers that still use model load terminology.
    def set_model_loading(self):
        self.set_model_starting()

    def set_model_loaded(self):
        self.set_model_running()

    def set_model_unloaded(self):
        self.set_model_stopped()

    def set_live_camera(self):
        self.view.input_file_label.setText("实时相机")
        self.view.input_file_label.setToolTip("使用已连接的实时事件相机")
        self._set_source_status(None)

    def set_input_file(self, file_path):
        self.view.input_file_label.setText(os.path.basename(file_path))
        self.view.input_file_label.setToolTip(file_path)
        self._set_source_status(file_path)

    def _set_source_status(self, file_path):
        set_source_status = getattr(self.view, "set_source_status", None)
        if callable(set_source_status):
            set_source_status(file_path)

    def set_weight_file(self, file_path):
        self.view.weight_path_label.setText(os.path.basename(file_path))
        self.view.weight_path_label.setToolTip(file_path)

    def _set_status(self, target, text, state):
        setter = getattr(self.view, "set_runtime_status", None)
        if callable(setter):
            setter(target, text, state)

    @staticmethod
    def _set_button_role(button, role):
        button.setProperty("buttonRole", role)
        style = getattr(button, "style", None)
        if not callable(style):
            return
        widget_style = style()
        widget_style.unpolish(button)
        widget_style.polish(button)
