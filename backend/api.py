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
    playback_progress_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera = CameraService(
            self.camera_queue,
            self.image_signal,
            self.camera_status_signal,
            self.playback_finished_signal,
            self.playback_progress_signal,
        )
        self.inference = InferenceService(self.camera_queue, self.prediction_signal)
        self.prediction_mode = "center"

    def is_camera_running(self):
        return self.camera.is_running()

    def is_inference_running(self):
        return self.inference.is_running()

    def is_recording(self):
        return self.camera.is_recording()

    def set_input_file(self, file_path):
        self.camera.set_input_file(file_path)

    def start_camera(self, config):
        self.camera.start(config=config)
        self._enqueue_camera_config()

    def restart_camera(self, config=None):
        self.camera.restart(config=config)
        self._enqueue_camera_config()

    def seek_playback(self, seek_fraction):
        if not self.camera.is_running():
            return
        self.camera.seek(seek_fraction)
        self._enqueue_camera_config()

    def update_playback_config(self, config):
        restarted = self.camera.apply_config(config)
        LOGGER.info(
            "Playback config updated: palette=%s, fps=%s, speed=%sx, roi=%s, noise=%s/%sus",
            config.palette,
            config.fps,
            config.replay_factor,
            config.roi,
            config.noise_filter_type,
            config.noise_filter_threshold_us,
        )
        return restarted

    def stop_camera(self):
        self.camera.stop(emit_finished=False)

    def update_camera_roi(self, roi):
        config = self.camera.current_config().with_updates(roi=roi)
        return self.update_playback_config(config)

    def set_prediction_mode(self, mode):
        mode_changed = self.prediction_mode != mode
        self.prediction_mode = mode
        LOGGER.info("Prediction mode set to: %s", mode)
        if mode_changed and self.is_inference_running() and self.inference.weights_path:
            self.restart_eventmamba()

    def set_noise_filter(self, filter_type, threshold_us, restart_camera=False):
        config = self.camera.current_config().with_updates(
            noise_filter_type=filter_type,
            noise_filter_threshold_us=threshold_us,
        )
        return self.update_playback_config(config)

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
