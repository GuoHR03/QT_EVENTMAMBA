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
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

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
REPLAY_SPEEDS = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
}
PLAYBACK_SLIDER_MAX = 10000


def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕获到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    try:
        input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    except EOFError:
        pass
    sys.exit(1)


sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(app_resource_path("form.ui"), self)
        self._init_playback_progress_ui()
        apply_app_theme(self)
        self.resize(1180, 780)

        self.settings = AppSettings()
        self.controller = AppController(self.settings)
        self.view_state = MainViewState(self)
        self.predictions = PredictionState(interval_ms=20)
        self._is_dragging_progress = False
        self._progress_total_us = 0

        self._connect_signals()
        self._init_view_state()

    def _init_playback_progress_ui(self):
        self.playback_progress_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.playback_progress_slider.setObjectName("playback_progress_slider")
        self.playback_progress_slider.setRange(0, PLAYBACK_SLIDER_MAX)
        self.playback_progress_slider.setTracking(False)
        self.playback_progress_slider.setEnabled(False)

        self.playback_time_label = QLabel("--:-- / --:--", self)
        self.playback_time_label.setObjectName("playback_time_label")
        self.playback_time_label.setMinimumWidth(138)
        self.playback_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        image_item = self.content_horizontal_layout.takeAt(0)
        image_widget = image_item.widget()
        viewer_widget = QWidget(self)
        viewer_widget.setObjectName("viewer_widget")
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        viewer_layout.addWidget(image_widget, 1)

        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        progress_layout.addWidget(self.playback_progress_slider, 1)
        progress_layout.addWidget(self.playback_time_label)
        viewer_layout.addLayout(progress_layout)

        self.content_horizontal_layout.insertWidget(0, viewer_widget, 1)
        self.content_horizontal_layout.setStretch(0, 1)
        self.content_horizontal_layout.setStretch(1, 0)

    def _connect_signals(self):
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.roi_window_button.clicked.connect(self.show_roi_window)
        self.palette_combo_box.currentTextChanged.connect(self.update_display_settings)
        self.fps_spin_box.valueChanged.connect(self.update_display_settings)
        self.replay_speed_combo_box.currentTextChanged.connect(self.update_replay_speed)
        self.select_weight_button.clicked.connect(self.select_weight_file)
        self.load_model_button.clicked.connect(self.load_eventmamba)
        self.unload_model_button.clicked.connect(self.unload_eventmamba)
        self.select_input_file_button.clicked.connect(self.select_input_file)
        self.playback_progress_slider.sliderPressed.connect(self._begin_progress_drag)
        self.playback_progress_slider.sliderMoved.connect(self._preview_progress_drag)
        self.playback_progress_slider.sliderReleased.connect(self._finish_progress_drag)
        self.controller.connect_view(
            self._display_image_with_prediction,
            self.log_text_edit.append,
            self._buffer_prediction_result,
            self.handle_playback_finished,
            self.handle_playback_progress,
        )

    def _init_view_state(self):
        self.log_text_edit.document().setMaximumBlockCount(500)
        self.replay_speed_combo_box.setCurrentText("1x")
        self.weight_path_label.setToolTip(self.weight_path_label.text())
        self.input_file_label.setToolTip(self.input_file_label.text())
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.view_state.set_model_unloaded()
        self._reset_playback_progress()

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

    def update_replay_speed(self):
        self._sync_capture_settings_from_ui()
        self.controller.update_replay_factor()

    def update_display_settings(self):
        self._sync_capture_settings_from_ui()
        self.controller.update_display_settings()

    def handle_playback_finished(self):
        self.stop_camera()

    def handle_playback_progress(self, current_us, total_us):
        total_us = max(0, int(total_us or 0))
        current_us = max(0, int(current_us or 0))
        if total_us > 0:
            current_us = min(total_us, current_us)
        self._progress_total_us = total_us
        self.playback_progress_slider.setEnabled(total_us > 0 and bool(self.controller.input_file_path))

        if not self._is_dragging_progress:
            value = 0
            if total_us > 0:
                value = int((current_us / total_us) * PLAYBACK_SLIDER_MAX)
            self.playback_progress_slider.blockSignals(True)
            self.playback_progress_slider.setValue(value)
            self.playback_progress_slider.blockSignals(False)

        if total_us > 0:
            total_label = _format_playback_time_us(total_us)
        else:
            total_label = "--:--"
        self.playback_time_label.setText(f"{_format_playback_time_us(current_us)} / {total_label}")

    def stop_camera(self):
        self.controller.stop_camera()
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.camera_image_label.setText("相机未启动")
        self.predictions.clear()
        self._reset_playback_progress()

    def select_input_file(self):
        file_path = choose_input_file(self)
        if not file_path:
            return

        self.view_state.set_input_file(file_path)
        self._reset_playback_progress()
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
            self._selected_replay_factor(),
        )

    def _selected_replay_factor(self):
        return REPLAY_SPEEDS.get(self.replay_speed_combo_box.currentText(), 1.0)

    def _begin_progress_drag(self):
        if self._progress_total_us <= 0:
            return
        self._is_dragging_progress = True

    def _preview_progress_drag(self, value):
        if self._progress_total_us <= 0:
            return
        value = max(0, min(PLAYBACK_SLIDER_MAX, int(value)))
        current_us = int((value / PLAYBACK_SLIDER_MAX) * self._progress_total_us)
        self.playback_time_label.setText(
            f"{_format_playback_time_us(current_us)} / {_format_playback_time_us(self._progress_total_us)}"
        )

    def _finish_progress_drag(self):
        if self._progress_total_us <= 0:
            self._is_dragging_progress = False
            return

        target_value = self.playback_progress_slider.sliderPosition()
        target_value = max(0, min(PLAYBACK_SLIDER_MAX, int(target_value)))

        self._is_dragging_progress = False
        self.playback_progress_slider.blockSignals(True)
        self.playback_progress_slider.setValue(target_value)
        self.playback_progress_slider.blockSignals(False)
        seek_fraction = target_value / PLAYBACK_SLIDER_MAX
        self._sync_capture_settings_from_ui()
        self.predictions.clear()
        self.controller.seek_playback(seek_fraction)

    def _reset_playback_progress(self):
        self._progress_total_us = 0
        self._is_dragging_progress = False
        self.playback_progress_slider.blockSignals(True)
        self.playback_progress_slider.setValue(0)
        self.playback_progress_slider.blockSignals(False)
        self.playback_progress_slider.setEnabled(False)
        self.playback_time_label.setText("--:-- / --:--")


def _format_playback_time_us(timestamp_us):
    timestamp_us = max(0, int(timestamp_us or 0))
    seconds = timestamp_us / 1_000_000.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    second_value = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{second_value:06.3f}"
    return f"{minutes:02d}:{second_value:06.3f}"


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
