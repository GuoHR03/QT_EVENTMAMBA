import sys
import os
import traceback
import ast
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import QFileDialog
from PyQt6 import uic
from choose_windows import choose_Window
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

        #信号与槽
        self.start_btn.clicked.connect(self.toggle_camera)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.ROI_btn.clicked.connect(self.choose_windowShow)
        self.combo_palette.currentTextChanged.connect(self.change_camera)
        self.fpsEdit.valueChanged.connect(self.change_camera)
        self.select_pt_btn.clicked.connect(self.choose_file_pt)
        self.apply_pt_btn.clicked.connect(self.load_Eventmamba)
        self.close_pt_btn.clicked.connect(self.unload_Eventmamba)
        self.select_file_btn.clicked.connect(self.choose_file)

        # 变量
        self.backend = BackendAPI()
        self.backend.image_signal.connect(self._display_image_with_prediction)
        self.backend.prediction_signal.connect(self._buffer_prediction_result)
        self.backend.playback_finished_signal.connect(self.on_playback_finished)
        self.file_path = None
        self.pt_path = None
        self.last_pred = None
        self.last_pred_mode = None
        self.prediction_buffer = {}
        self.nn_interval_ms = 20

    def toggle_camera(self):
        """相机启动链接的槽"""
        if not self.backend.is_camera_running():
            self.backend.start_camera(self.combo_palette.currentText(), self.fpsEdit.value())
            self.start_btn.setText("停止相机")
            self.record_btn.setEnabled(True)
        else:
            """关闭相机"""
            self.stop_camera_logic()

    def toggle_recording(self):
        if self.backend.is_camera_running() and self.backend.camera_thread:
            if not self.backend.camera_thread.is_recording:
                self.backend.start_recording()
                self.record_btn.setText("停止录制")
                self.record_btn.setStyleSheet("background-color: red; color: white;")
            else:
                self.backend.stop_recording()
                self.record_btn.setText("开始录制")
                self.record_btn.setStyleSheet("")

    def _display_image_with_prediction(self, cv_img, img_timestamp):
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
        for ts_key in sorted(self.prediction_buffer.keys()):
            if frame_start_time <= ts_key <= frame_end_time:
                matched_pred, matched_mode, matched_cropped = self.prediction_buffer[ts_key]
                break
        if matched_pred is None:
            for ts_key in sorted(self.prediction_buffer.keys()):
                if ts_key <= frame_end_time:
                    matched_pred, matched_mode, matched_cropped = self.prediction_buffer[ts_key]
                else:
                    break

        for ts_key in list(self.prediction_buffer.keys()):
            if ts_key < frame_end_time - 200000:
                del self.prediction_buffer[ts_key]

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)
        if matched_pred is not None:
            px, py = self._map_pred_to_pixel_crop(matched_pred, width, height)
            #px, py = self._map_pred_to_pixel_norm(matched_pred, width, height)
            if px is not None and py is not None:
                painter = QPainter(q_img)
                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(QColor(255, 0, 0, 80))
                #painter.drawEllipse(px - 8, py - 8, 16, 16)
                painter.drawEllipse(int(px) - 8, int(py) - 8, 16, 16)
                painter.end()

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def _buffer_prediction_result(self, result, pred_timestamp):
        self.textEdit.append(result)

        if not isinstance(result, str):
            return
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
        self.last_pred = pred_data

    def closeEvent(self, event):
        """关闭程序"""
        self.backend.close()
        event.accept()

    def change_camera(self):
        """更改相机参数"""
        if self.backend.is_camera_running():
            self.toggle_camera() # 停止旧相机
            QApplication.processEvents()
            self.toggle_camera() # 带着新选的颜色重新启动相机

    #  从这里继续查代码
    def on_playback_finished(self):
        """文件播放完后的自动处理"""
        self.stop_camera_logic()

    def stop_camera_logic(self):
        """抽离出来的停止逻辑"""
        self.backend.stop_camera()

        self.start_btn.setText("启动相机")
        self.record_btn.setEnabled(False)
        self.image_label.setText("相机未启动")
        self.current_cam_size = None

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择离线视频文件",
            "",
            "视频(*.raw *.hdf5 *.h5 *.aedat4);;所有文件 (*)"
        )
        if file_path:
            self.file_path = file_path
            self.backend.set_input_file(file_path)
            self.file_path_label.setText(file_path)
            # 修改
            self.file_path_label.setText(os.path.basename(file_path))
            if self.backend.is_camera_running():
                self.toggle_camera()

    def choose_file_pt(self):
            pt_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择权重文件",
                "",
                "pth (*.pth);;所有文件 (*)"
            )
            if pt_path:
                self.pt_path = pt_path
                self.pt_path_label.setText(pt_path)
                # 修改
                self.pt_path_label.setText(os.path.basename(pt_path))
                if self.backend.is_camera_running():
                    self.toggle_camera()

    def load_Eventmamba(self):
        """加载Eventmamba模型以及网络通信"""
        if self.pt_path == None:
            print("没有加载权重")
        else:
            self.backend.start_eventmamba(self.pt_path)
        self.apply_pt_btn.setEnabled(False)
        self.close_pt_btn.setEnabled(True)

    def unload_Eventmamba(self):
        """关闭Eventmamba模型 以及WSL"""
        print("关闭权重WSL")
        self.backend.stop_eventmamba()
        self.apply_pt_btn.setEnabled(True)
        self.close_pt_btn.setEnabled(False)
        self.last_pred = None
        self.last_pred_mode = None

    def _map_pred_to_pixel_crop(self, pred, width, height):
        if pred is None:
            return None, None
        x, y = pred
        px = int(x * 512 + 96)
        py = int(y * 512 - 16)
        px = int(px * 2)
        py = int(py * 1.5)
        if px < 0 or py < 0 or px >= width or py >= height:
            return None, None
        return px, py

    def _map_pred_to_pixel_norm(self, pred, width, height):
        if pred is None:
            return None, None
        x, y = pred
        px = int(x * width)
        py = int(y * height)
        if px < 0 or py < 0 or px >= width or py >= height:
            return None, None
        return px, py

    def choose_windowShow(self):
        if hasattr(self, 'new_window') and self.new_window is not None:
            self.new_window.close()
            self.new_window.deleteLater()
        self.new_window = choose_Window()
        self.new_window.roi_applied.connect(self.on_roi_applied)
        self.new_window.show()

    def on_roi_applied(self, roi_x, roi_y, roi_width, roi_height):
        print(f"[ROI] 获取到参数: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
        self.backend.set_roi(roi_x, roi_y, roi_width, roi_height)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
