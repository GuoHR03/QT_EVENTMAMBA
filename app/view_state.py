import os


class MainViewState:
    def __init__(self, view):
        self.view = view

    def set_camera_running(self):
        controller = getattr(self.view, "controller", None)
        source_is_file = bool(getattr(controller, "input_file_path", None))
        self.view.start_camera_button.setText("停止播放" if source_is_file else "停止相机")
        self._set_button_role(self.view.start_camera_button, "primary")
        self._set_status("camera", "运行中", "active")
        self.set_recording_stopped(enabled=True)

    def set_camera_stopped(self):
        controller = getattr(self.view, "controller", None)
        source_is_file = bool(getattr(controller, "input_file_path", None))
        self.view.start_camera_button.setText("开始播放" if source_is_file else "启动相机")
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

    def set_model_loading(self):
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(False)
        self.view.load_model_button.setText("正在加载...")
        self._set_status("model", "模型加载中", "pending")

    def set_model_loaded(self):
        self.view.load_model_button.setText("已加载")
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(True)
        self.view.select_weight_button.setEnabled(True)
        self._set_status("model", "模型已加载", "active")

    def set_model_unloaded(self):
        self.view.load_model_button.setEnabled(True)
        self.view.unload_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(True)
        self.view.load_model_button.setText("加载模型")
        self._set_button_role(self.view.load_model_button, "primary")
        self._set_status("model", "模型未加载", "idle")

    def set_input_file(self, file_path):
        self.view.input_file_label.setText(os.path.basename(file_path))
        self.view.input_file_label.setToolTip(file_path)
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
