import sys
import traceback

try:
    from .bootstrap import app_resource_path, configure_runtime
except ImportError:
    from bootstrap import app_resource_path, configure_runtime

configure_runtime(__file__)

from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

try:
    from .choose_windows import ChooseWindow
    from .controller import AppController
    from .file_dialogs import choose_input_file, choose_weights_file
    from .log_formatter import backend_message, mode_display_name, noise_settings_message, roi_settings_message
    from .prediction_overlay import draw_prediction
    from .prediction_state import PredictionState
    from .settings import AppSettings
    from .theme import apply_app_theme
    from .view_state import MainViewState
except ImportError:
    from choose_windows import ChooseWindow
    from controller import AppController
    from file_dialogs import choose_input_file, choose_weights_file
    from log_formatter import backend_message, mode_display_name, noise_settings_message, roi_settings_message
    from prediction_overlay import draw_prediction
    from prediction_state import PredictionState
    from settings import AppSettings
    from theme import apply_app_theme
    from view_state import MainViewState

SUPPORTED_PALETTES = {"Dark", "Light", "CoolWarm", "Gray"}


def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕获到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    sys.exit(1)


sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(app_resource_path("form.ui"), self)
        apply_app_theme(self)
        self.resize(1180, 780)

        self.settings = AppSettings()
        self.controller = AppController(self.settings)
        self.view_state = MainViewState(self)
        self.predictions = PredictionState(interval_ms=20)

        self._connect_signals()
        self._init_view_state()

    def _connect_signals(self):
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.roi_window_button.clicked.connect(self.show_roi_window)
        self.palette_combo_box.currentTextChanged.connect(self.restart_camera_if_running)
        self.fps_spin_box.valueChanged.connect(self.restart_camera_if_running)
        self.select_weight_button.clicked.connect(self.select_weight_file)
        self.load_model_button.clicked.connect(self.load_eventmamba)
        self.unload_model_button.clicked.connect(self.unload_eventmamba)
        self.select_input_file_button.clicked.connect(self.select_input_file)
        self.controller.connect_view(
            self._display_image_with_prediction,
            self.log_text_edit.append,
            self._buffer_prediction_result,
            self.handle_playback_finished,
        )

    def _init_view_state(self):
        self.log_text_edit.document().setMaximumBlockCount(500)
        self.weight_path_label.setToolTip(self.weight_path_label.text())
        self.input_file_label.setToolTip(self.input_file_label.text())
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.view_state.set_model_unloaded()

    def toggle_camera(self):
        if not self.controller.is_camera_running():
            self._sync_capture_settings_from_ui()
            self.controller.start_camera()
            self.view_state.set_camera_running()
        else:
            self.stop_camera()

    def toggle_recording(self):
        recording_started = self.controller.toggle_recording()
        if recording_started is None:
            return
        if recording_started:
            self.view_state.set_recording_running()
        else:
            self.view_state.set_recording_stopped(enabled=True)

    def _display_image_with_prediction(self, cv_img, img_timestamp):
        if hasattr(cv_img, "flags") and not cv_img.flags["C_CONTIGUOUS"]:
            cv_img = cv_img.copy()

        if len(cv_img.shape) == 3:
            height, width, channel = cv_img.shape
            bytes_per_line = channel * width
            img_format = QImage.Format.Format_BGR888
        else:
            height, width = cv_img.shape
            bytes_per_line = width
            img_format = QImage.Format.Format_Grayscale8

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)
        matched_prediction = self.predictions.match_frame(img_timestamp)
        if matched_prediction is not None:
            draw_prediction(q_img, matched_prediction, width, height, self.settings.roi)

        pixmap = QPixmap.fromImage(q_img)
        self.camera_image_label.setPixmap(
            pixmap.scaled(
                self.camera_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _buffer_prediction_result(self, result, pred_timestamp):
        self.log_text_edit.append(backend_message(result))
        self.predictions.add_result(result, pred_timestamp, self.settings.prediction_mode)

    def closeEvent(self, event):
        self.controller.close()
        event.accept()

    def restart_camera_if_running(self):
        if not self.controller.is_camera_running():
            return
        QApplication.processEvents()
        self._sync_capture_settings_from_ui()
        self.controller.restart_camera_if_running()

    def handle_playback_finished(self):
        self.stop_camera()

    def stop_camera(self):
        self.controller.stop_camera()
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.camera_image_label.setText("相机未启动")
        self.predictions.clear()

    def select_input_file(self):
        file_path = choose_input_file(self)
        if not file_path:
            return

        self.view_state.set_input_file(file_path)
        QApplication.processEvents()
        self._sync_capture_settings_from_ui()
        self.controller.set_input_file(file_path, restart_if_running=True)

    def select_weight_file(self):
        weights_path = choose_weights_file(self)
        if not weights_path:
            return

        self.view_state.set_weight_file(weights_path)
        stopped_running_model = self.controller.set_weights_path(weights_path)
        if stopped_running_model:
            self.view_state.set_model_unloaded()
            self.predictions.clear()
            self.log_text_edit.append("已选择新权重，请重新加载模型")

    def load_eventmamba(self):
        if self.controller.weights_path is None:
            self.log_text_edit.append("请先选择权重文件")
            return

        self.view_state.set_model_loading()
        self.log_text_edit.append(
            f"正在启动 WSL 推理服务并加载{mode_display_name(self.settings.prediction_mode)}模式权重，首次加载可能需要几秒钟..."
        )
        QApplication.processEvents()

        try:
            self.controller.load_model()
        except Exception as exc:
            self.view_state.set_model_unloaded()
            self.log_text_edit.append(f"加载模型失败：{exc}")
            return

        self.log_text_edit.append("WSL 推理服务已就绪，权重加载完成")
        self.view_state.set_model_loaded()

    def unload_eventmamba(self):
        self.controller.unload_model()
        self.view_state.set_model_unloaded()
        self.log_text_edit.append("WSL 推理服务已关闭，可以重新选择权重并加载")
        self.predictions.clear()

    def show_roi_window(self):
        self.roi_window = ChooseWindow(
            initial_mode=self.settings.prediction_mode,
            initial_roi=self.settings.roi,
            initial_noise_filter_type=self.settings.noise_filter_type,
            initial_noise_filter_threshold_us=self.settings.noise_filter_threshold_us,
        )
        self.roi_window.settings_confirmed.connect(self.on_settings_confirmed)
        self.roi_window.show()

    def on_settings_confirmed(self, roi, mode, filter_type, threshold_us):
        self._sync_capture_settings_from_ui()
        camera_settings_changed = self.controller.apply_settings(
            roi,
            mode,
            filter_type,
            threshold_us,
        )
        if camera_settings_changed and self.controller.is_camera_running():
            QApplication.processEvents()

        self.log_text_edit.append(noise_settings_message(filter_type, self.settings.noise_filter_threshold_us))
        if roi is not None:
            self.log_text_edit.append(roi_settings_message(self.settings.roi, mode))

    def _selected_palette(self):
        selected = self.palette_combo_box.currentText()
        if selected in SUPPORTED_PALETTES:
            return selected
        return "Dark"

    def _sync_capture_settings_from_ui(self):
        self.controller.sync_capture_settings(
            self._selected_palette(),
            self.fps_spin_box.value(),
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
