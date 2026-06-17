import os


class MainViewState:
    def __init__(self, view):
        self.view = view

    def set_camera_running(self):
        self.view.start_camera_button.setText("停止相机")
        self.set_recording_stopped(enabled=True)

    def set_camera_stopped(self):
        self.view.start_camera_button.setText("启动相机")

    def set_recording_running(self):
        self.view.record_button.setEnabled(True)
        self.view.record_button.setText("停止录制")
        self.view.record_button.setStyleSheet("background-color: red; color: white;")

    def set_recording_stopped(self, enabled):
        self.view.record_button.setEnabled(enabled)
        self.view.record_button.setText("开始录制")
        self.view.record_button.setStyleSheet("")

    def set_model_loading(self):
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(False)
        self.view.load_model_button.setText("正在加载...")

    def set_model_loaded(self):
        self.view.load_model_button.setText("已加载")
        self.view.load_model_button.setEnabled(False)
        self.view.unload_model_button.setEnabled(True)
        self.view.select_weight_button.setEnabled(True)

    def set_model_unloaded(self):
        self.view.load_model_button.setEnabled(True)
        self.view.unload_model_button.setEnabled(False)
        self.view.select_weight_button.setEnabled(True)
        self.view.load_model_button.setText("加载模型")

    def set_input_file(self, file_path):
        self.view.input_file_label.setText(os.path.basename(file_path))
        self.view.input_file_label.setToolTip(file_path)

    def set_weight_file(self, file_path):
        self.view.weight_path_label.setText(os.path.basename(file_path))
        self.view.weight_path_label.setToolTip(file_path)
