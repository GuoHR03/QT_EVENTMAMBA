import queue
import logging

from PyQt6.QtCore import QObject, pyqtSignal

from backend.camera_service import CameraService
from backend.inference_service import InferenceService

LOGGER = logging.getLogger(__name__)


class BackendAPI(QObject):
    image_signal = pyqtSignal(object, int)
    camera_status_signal = pyqtSignal(str)
    prediction_signal = pyqtSignal(object, int)
    playback_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera = CameraService(
            self.camera_queue,
            self.image_signal,
            self.camera_status_signal,
            self.playback_finished_signal,
        )
        self.inference = InferenceService(self.camera_queue, self.prediction_signal)
        self.prediction_mode = "center"
        self.noise_filter_type = "none"
        self.noise_filter_threshold_us = 10000

    def is_camera_running(self):
        return self.camera.is_running()

    def is_inference_running(self):
        return self.inference.is_running()

    def is_recording(self):
        return self.camera.is_recording()

    def set_input_file(self, file_path):
        self.camera.set_input_file(file_path)

    def start_camera(self, palette, fps, roi=None):
        self.camera.start(
            palette,
            fps,
            roi,
            self.noise_filter_type,
            self.noise_filter_threshold_us,
        )
        self._enqueue_camera_config()

    def restart_camera(self, palette=None, fps=None, roi=None):
        self.camera.restart(
            palette,
            fps,
            roi,
            self.noise_filter_type,
            self.noise_filter_threshold_us,
        )
        self._enqueue_camera_config()

    def stop_camera(self):
        self.camera.stop()

    def update_camera_roi(self, roi):
        if self.camera.is_running():
            self.restart_camera(roi=roi)
        else:
            LOGGER.info("Camera is not running; ROI will be applied on next start")

    def set_prediction_mode(self, mode):
        mode_changed = self.prediction_mode != mode
        self.prediction_mode = mode
        LOGGER.info("Prediction mode set to: %s", mode)
        if mode_changed and self.is_inference_running() and self.inference.weights_path:
            self.restart_eventmamba()

    def set_noise_filter(self, filter_type, threshold_us, restart_camera=True):
        filter_type = filter_type or "none"
        try:
            threshold_us = int(threshold_us)
        except (TypeError, ValueError):
            threshold_us = 10000
        threshold_us = max(1, threshold_us)

        changed = (
            self.noise_filter_type != filter_type
            or self.noise_filter_threshold_us != threshold_us
        )
        self.noise_filter_type = filter_type
        self.noise_filter_threshold_us = threshold_us
        LOGGER.info("Noise filter: %s, threshold=%sus", filter_type, threshold_us)

        if changed and restart_camera and self.camera.is_running():
            self.restart_camera(roi=self.camera.current_roi())

    def start_recording(self):
        self.camera.start_recording()

    def stop_recording(self):
        self.camera.stop_recording()

    def start_eventmamba(self, weights_path, port=5555, host="127.0.0.1"):
        self.inference.start(weights_path, self.prediction_mode, port=port, host=host)
        self._enqueue_camera_config()

    def restart_eventmamba(self, port=5555, host="127.0.0.1"):
        self.inference.restart(self.prediction_mode, port=port, host=host)
        self._enqueue_camera_config()

    def stop_eventmamba(self):
        self.inference.stop()

    def close(self):
        self.stop_camera()
        self.stop_eventmamba()

    def _enqueue_camera_config(self):
        if not self.camera.is_running():
            return
        if self.prediction_mode not in ("center", "ellipse"):
            return
        camera_size = self.camera.current_size()
        if camera_size is None:
            return
        width, height = camera_size
        payload = {
            "msg_type": "CONFIG",
            "width": width,
            "height": height,
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
