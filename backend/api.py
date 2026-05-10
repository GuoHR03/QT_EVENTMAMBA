import os
import queue
import subprocess
import time

import zmq
from PyQt6.QtCore import QObject, pyqtSignal

from backend.Camera import CameraThread
from backend.NetworkThread import NetworkThread


class BackendAPI(QObject):
    image_signal = pyqtSignal(object, int)
    prediction_signal = pyqtSignal(str, int)
    playback_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera_thread = None
        self.network_thread = None
        self.backend_process = None
        self.file_path = None
        self.weights_path = None
        self.prediction_mode = "center"
        self.wsl_distro = os.environ.get("EVENTMAMBA_WSL_DISTRO", "EventMamba_mini")
        self.linux_python = os.environ.get(
            "EVENTMAMBA_LINUX_PYTHON",
            "/opt/miniconda3/envs/eventmamba/bin/python",
        )

    def is_camera_running(self):
        return self.camera_thread is not None and self.camera_thread.isRunning()

    def is_inference_running(self):
        return self.network_thread is not None and self.network_thread.isRunning()

    def set_input_file(self, file_path):
        self.file_path = file_path

    def start_camera(self, palette, fps, roi=None):
        self._last_palette = palette
        self._last_fps = fps
        if self.camera_thread is not None:
            self.stop_camera()

        kwargs = {
            "palette_type": palette,
            "fps": fps,
            "target_queue": self.camera_queue,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path
        if roi:
            kwargs["roi"] = roi

        self.camera_thread = CameraThread(**kwargs)
        self.camera_thread.image_signal.connect(self.image_signal.emit)
        self.camera_thread.finished_signal.connect(self.playback_finished_signal.emit)
        self.camera_thread.start()
        self._enqueue_camera_config()

    def stop_camera(self):
        if not self.camera_thread:
            return

        self.camera_thread.stop()
        if not self.camera_thread.wait(1000):
            self.camera_thread.terminate()
            self.camera_thread.wait(500)

        self.camera_thread.deleteLater()
        self.camera_thread = None

    def update_camera_roi(self, roi):
        """动态更新相机 ROI 参数；如果相机运行中则重启应用。"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
            self.start_camera_with_roi(roi)
        else:
            print("[BackendAPI] 相机未运行，ROI 将在下次启动时生效")

    def set_prediction_mode(self, mode):
        """设置预测模式 (center 或 ellipse)。"""
        mode_changed = self.prediction_mode != mode
        self.prediction_mode = mode
        print(f"[BackendAPI] 预测模式已设置为: {mode}")
        if mode_changed and self.is_inference_running() and self.weights_path:
            self.restart_eventmamba()

    def start_camera_with_roi(self, roi):
        """内部方法：使用指定 ROI 启动相机。"""
        kwargs = {
            "palette_type": self._last_palette if hasattr(self, "_last_palette") else "Dark",
            "fps": self._last_fps if hasattr(self, "_last_fps") else 30,
            "target_queue": self.camera_queue,
            "roi": roi,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path

        self.camera_thread = CameraThread(**kwargs)
        self.camera_thread.image_signal.connect(self.image_signal.emit)
        self.camera_thread.finished_signal.connect(self.playback_finished_signal.emit)
        self.camera_thread.start()
        self._enqueue_camera_config()

    def start_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.start_recording()

    def stop_recording(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_recording()

    def start_eventmamba(self, weights_path, port=5555, host="127.0.0.1"):
        if not weights_path:
            raise ValueError("weights_path is required")
        if self.prediction_mode not in ("center", "ellipse"):
            raise ValueError("prediction_mode is not set")

        self.weights_path = weights_path
        wsl_weights_path = self._to_wsl_path(weights_path)

        if self.backend_process is None or self.backend_process.poll() is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            linux_script = "linux_backend.py"
            weights_arg = "--ellipse-weights" if self.prediction_mode == "ellipse" else "--center-weights"
            cmd = [
                "wsl",
                "-d",
                self.wsl_distro,
                self.linux_python,
                linux_script,
                weights_arg,
                wsl_weights_path,
                "--port",
                str(port),
            ]
            self.backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=project_dir,
            )
            self._wait_for_backend_ready(host=host, port=port)

        if self.network_thread is None or not self.network_thread.isRunning():
            self.network_thread = NetworkThread(self.camera_queue, host=host, port=port, request_timeout_ms=1000)
            self.network_thread.result_signal.connect(self.prediction_signal.emit)
            self.network_thread.start()
        self._enqueue_camera_config()

    def restart_eventmamba(self, port=5555, host="127.0.0.1"):
        if not self.weights_path:
            return
        self.stop_eventmamba()
        self.start_eventmamba(self.weights_path, port=port, host=host)

    def stop_eventmamba(self):
        if self.network_thread:
            self.network_thread.stop()
            if not self.network_thread.wait(2000):
                self.network_thread.terminate()
                self.network_thread.wait(500)
            self.network_thread.deleteLater()
            self.network_thread = None

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=2)
            except Exception:
                try:
                    self.backend_process.kill()
                except Exception:
                    pass
            self.backend_process = None

        try:
            subprocess.run(
                ["wsl", "-d", self.wsl_distro, "pkill", "-f", "linux_backend.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            pass

    def close(self):
        self.stop_camera()
        self.stop_eventmamba()

    def _wait_for_backend_ready(self, host, port, timeout_s=10.0, poll_interval_s=0.2):
        endpoint = f"tcp://{host}:{port}"
        deadline = time.time() + timeout_s
        context = zmq.Context()
        try:
            while time.time() < deadline:
                if self.backend_process is not None and self.backend_process.poll() is not None:
                    raise RuntimeError("WSL 推理服务启动失败，进程已退出")

                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, int(poll_interval_s * 1000))
                socket.setsockopt(zmq.SNDTIMEO, int(poll_interval_s * 1000))
                try:
                    socket.connect(endpoint)
                    socket.send_pyobj({"msg_type": "PING"})
                    reply = socket.recv_string()
                    if reply == "READY":
                        return
                except zmq.Again:
                    time.sleep(poll_interval_s)
                except zmq.ZMQError:
                    time.sleep(poll_interval_s)
                finally:
                    socket.close(linger=0)

            raise TimeoutError("等待 WSL 推理服务就绪超时")
        finally:
            context.term()

    def _enqueue_camera_config(self):
        if not self.camera_thread or not self.camera_thread.isRunning():
            return
        if self.prediction_mode not in ("center", "ellipse"):
            return
        payload = {
            "msg_type": "CONFIG",
            "width": self.camera_thread.width,
            "height": self.camera_thread.height,
            "prediction_mode": self.prediction_mode,
        }
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.camera_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _to_wsl_path(self, path):
        if not path:
            return path
        if path.startswith("\\\\wsl$\\"):
            parts = path.split("\\")
            if len(parts) >= 4:
                distro = parts[2]
                inner = "/".join(parts[3:])
                if distro == self.wsl_distro:
                    return f"/{inner}".replace("\\", "/")
                return f"/mnt/wsl/{distro}/{inner}".replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            rest = path[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
        return path
