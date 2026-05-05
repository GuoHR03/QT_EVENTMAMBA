import sys
import os
import traceback
import ast
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import QFileDialog
from PyQt6 import uic
from choose_windows import ChooseWindow
base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))
if os.path.isdir(os.path.join(root_dir, "backend")) and root_dir not in sys.path:
    sys.path.insert(0, root_dir)

sdk_root = os.environ.get("METAVISION_SDK_PATH", "E:\\Metavision\\Prophesee")
extra_dll_dirs = [
    os.path.join(root_dir, "libs", "bin"),
    os.path.join(sdk_root, "bin"),
    os.path.join(sdk_root, "third_party", "bin"),
    os.path.join(sdk_root, "lib", "hdf5", "plugin"),
]
for dll_dir in extra_dll_dirs:
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

from backend.api import BackendAPI


def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕捉到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    sys.exit(1)

sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(base_dir, "form.ui")
        uic.loadUi(ui_path, self)

        # 信号与槽
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.roi_window_button.clicked.connect(self.show_roi_window)
        self.palette_combo_box.currentTextChanged.connect(self.restart_camera_if_running)
        self.fps_spin_box.valueChanged.connect(self.restart_camera_if_running)
        self.select_weight_button.clicked.connect(self.select_weight_file)
        self.load_model_button.clicked.connect(self.load_eventmamba)
        self.unload_model_button.clicked.connect(self.unload_eventmamba)
        self.select_input_file_button.clicked.connect(self.select_input_file)
        self.unload_model_button.setEnabled(False)
        self.log_text_edit.document().setMaximumBlockCount(500)

        # 变量
        self.backend = BackendAPI()
        self.backend.image_signal.connect(self._display_image_with_prediction)
        self.backend.prediction_signal.connect(self._buffer_prediction_result)
        self.backend.playback_finished_signal.connect(self.handle_playback_finished)
        self.input_file_path = None
        self.weight_path = None
        self.last_prediction = None
        self.last_prediction_mode = None
        self.prediction_buffer = {}
        self.nn_interval_ms = 20

    def toggle_camera(self):
        """切换相机或离线文件播放状态。"""
        if not self.backend.is_camera_running():
            self.backend.start_camera(self.palette_combo_box.currentText(), self.fps_spin_box.value())
            self.start_camera_button.setText("停止相机")
            self.record_button.setEnabled(True)
        else:
            self.stop_camera()

    def toggle_recording(self):
        if self.backend.is_camera_running() and self.backend.camera_thread:
            if not self.backend.camera_thread.is_recording:
                self.backend.start_recording()
                self.record_button.setText("停止录制")
                self.record_button.setStyleSheet("background-color: red; color: white;")
            else:
                self.backend.stop_recording()
                self.record_button.setText("开始录制")
                self.record_button.setStyleSheet("")

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

        frame_end_time = img_timestamp
        frame_start_time = frame_end_time - int(self.nn_interval_ms * 1000)

        matched_pred = None
        matched_mode = None
        matched_cropped = False
        sorted_ts = sorted(self.prediction_buffer.keys())
        for ts_key in sorted_ts:
            if frame_start_time <= ts_key <= frame_end_time:
                matched_pred, matched_mode, matched_cropped = self.prediction_buffer[ts_key]
                break
        if matched_pred is None:
            for ts_key in sorted_ts:
                if ts_key <= frame_end_time:
                    matched_pred, matched_mode, matched_cropped = self.prediction_buffer[ts_key]
                else:
                    break

        for ts_key in list(self.prediction_buffer.keys()):
            if ts_key < frame_end_time - 200000:
                del self.prediction_buffer[ts_key]

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)
        if matched_pred is not None:
            px, py = self._map_prediction_to_pixel(
                matched_pred, matched_mode, matched_cropped, width, height
            )
            if px is not None and py is not None:
                painter = QPainter(q_img)
                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(QColor(255, 0, 0, 80))
                painter.drawEllipse(px - 8, py - 8, 16, 16)
                painter.end()

        pixmap = QPixmap.fromImage(q_img)
        self.camera_image_label.setPixmap(pixmap.scaled(
            self.camera_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def _buffer_prediction_result(self, result, pred_timestamp):
        if not isinstance(result, str):
            self.log_text_edit.append(str(result))
            return
        self.log_text_edit.append(result)
        marker = "输出结果为："
        if marker not in result:
            return
        parts = result.split(marker, 1)[1].strip()
        is_cropped = False
        if "|cropped:" in parts:
            main_part, cropped_part = parts.rsplit("|cropped:", 1)
            is_cropped = cropped_part.strip().lower() == "true"
            payload = main_part.strip()
        else:
            payload = parts
        try:
            values = ast.literal_eval(payload)
        except Exception:
            return
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            return
        x, y = values[0], values[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return
        pred_data = (float(x), float(y))
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            pred_mode = "norm"
        else:
            pred_mode = "pixel"
        self.prediction_buffer[pred_timestamp] = (pred_data, pred_mode, is_cropped)
        self.last_prediction = pred_data
        self.last_prediction_mode = pred_mode

    def closeEvent(self, event):
        """关闭程序"""
        self.backend.close()
        event.accept()

    def restart_camera_if_running(self):
        """参数变化后重启当前播放源。"""
        if self.backend.is_camera_running():
            self.toggle_camera() # 停止旧相机
            QApplication.processEvents()
            self.toggle_camera() # 带着新选的颜色重新启动相机

    def handle_playback_finished(self):
        """文件播放完后的自动处理"""
        self.stop_camera()

    def stop_camera(self):
        """停止相机或离线文件播放并重置 UI 状态。"""
        self.backend.stop_camera()

        self.start_camera_button.setText("启动相机")
        self.record_button.setEnabled(False)
        self.record_button.setText("开始录制")
        self.record_button.setStyleSheet("")
        self.camera_image_label.setText("相机未启动")
        self.current_cam_size = None
        self.prediction_buffer.clear()

    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择离线视频文件",
            r"E:\Code\Qt\UI_Event-main\record",
            "视频(*.raw *.hdf5 *.h5 *.aedat4);;所有文件 (*)"
        )
        if file_path:
            self.input_file_path = file_path
            self.backend.set_input_file(file_path)
            self.input_file_label.setText(os.path.basename(file_path))
            if self.backend.is_camera_running():
                self.toggle_camera()

    def select_weight_file(self):
        pt_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择权重文件",
            r"E:\Code\Qt\UI_Event-main\checkpoint",
            "pth (*.pth);;所有文件 (*)"
        )
        if pt_path:
            self.weight_path = pt_path
            self.weight_path_label.setText(os.path.basename(pt_path))
            if self.backend.is_camera_running():
                self.toggle_camera()

    def load_eventmamba(self):
        """加载Eventmamba模型以及网络通信"""
        if self.weight_path is None:
            self.log_text_edit.append("请先选择权重文件")
            return

        try:
            self.backend.start_eventmamba(self.weight_path)
        except Exception as exc:
            self.log_text_edit.append(f"加载 EventMamba 失败: {exc}")
            return

        self.load_model_button.setEnabled(False)
        self.unload_model_button.setEnabled(True)

    def unload_eventmamba(self):
        """关闭Eventmamba模型 以及WSL"""
        print("关闭权重WSL")
        self.backend.stop_eventmamba()
        self.load_model_button.setEnabled(True)
        self.unload_model_button.setEnabled(False)
        self.last_prediction = None
        self.last_prediction_mode = None
        self.prediction_buffer.clear()

    def _map_prediction_to_pixel(self, pred, mode, is_cropped, width, height):
        if is_cropped:
            return self._map_cropped_prediction_to_pixel(pred, width, height)
        if mode == "norm":
            return self._map_normalized_prediction_to_pixel(pred, width, height)
        return self._map_raw_prediction_to_pixel(pred, width, height)

    def _map_cropped_prediction_to_pixel(self, pred, width, height):
        if pred is None:
            return None, None
        x, y = pred
        canonical_x = x * 512 + 96
        canonical_y = y * 512 - 16
        px = int(canonical_x * width / 640)
        py = int(canonical_y * height / 480)
        if px < 0 or py < 0 or px >= width or py >= height:
            return None, None
        return px, py

    def _map_normalized_prediction_to_pixel(self, pred, width, height):
        if pred is None:
            return None, None
        x, y = pred
        px = int(x * width)
        py = int(y * height)
        if px < 0 or py < 0 or px >= width or py >= height:
            return None, None
        return px, py

    def _map_raw_prediction_to_pixel(self, pred, width, height):
        if pred is None:
            return None, None
        x, y = pred
        px = int(x)
        py = int(y)
        if px < 0 or py < 0 or px >= width or py >= height:
            return None, None
        return px, py

    def show_roi_window(self):
        self.roi_window = ChooseWindow()
        self.roi_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
