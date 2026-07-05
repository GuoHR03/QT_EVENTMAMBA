from backend.Camera import CameraThread
from backend.settings import DEFAULT_FPS, DEFAULT_NOISE_FILTER_THRESHOLD_US


class CameraService:
    def __init__(self, frame_queue, image_signal, status_signal, finished_signal):
        self.frame_queue = frame_queue
        self.image_signal = image_signal
        self.status_signal = status_signal
        self.finished_signal = finished_signal
        self.thread = None
        self.file_path = None
        self.last_palette = "Dark"
        self.last_fps = DEFAULT_FPS
        self.last_replay_factor = 1.0

    def is_running(self):
        return self.thread is not None and self.thread.isRunning()

    def is_recording(self):
        return self.is_running() and self.thread.is_recording

    def set_input_file(self, file_path):
        self.file_path = file_path

    def start(
        self,
        palette,
        fps,
        roi=None,
        replay_factor=1.0,
        noise_filter_type="none",
        noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US,
    ):
        self.last_palette = palette
        self.last_fps = fps
        self.last_replay_factor = replay_factor
        if self.thread is not None:
            self.stop()

        kwargs = {
            "palette_type": palette,
            "fps": fps,
            "replay_factor": replay_factor,
            "target_queue": self.frame_queue,
            "noise_filter_type": noise_filter_type,
            "noise_filter_threshold_us": noise_filter_threshold_us,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path
        if roi:
            kwargs["roi"] = roi

        self.thread = CameraThread(**kwargs)
        self.thread.image_signal.connect(self.image_signal.emit)
        self.thread.status_signal.connect(self.status_signal.emit)
        self.thread.finished_signal.connect(self.finished_signal.emit)
        self.thread.start()

    def restart(
        self,
        palette=None,
        fps=None,
        roi=None,
        replay_factor=None,
        noise_filter_type="none",
        noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US,
    ):
        palette = palette if palette is not None else self.last_palette
        fps = fps if fps is not None else self.last_fps
        replay_factor = replay_factor if replay_factor is not None else self.last_replay_factor
        self.stop()
        self.start(palette, fps, roi, replay_factor, noise_filter_type, noise_filter_threshold_us)

    def set_replay_factor(self, replay_factor):
        self.last_replay_factor = replay_factor
        if self.thread is not None:
            self.thread.set_replay_factor(replay_factor)

    def set_display_settings(self, palette, fps):
        self.last_palette = palette
        self.last_fps = fps
        if self.thread is not None:
            self.thread.set_display_settings(palette, fps)

    def stop(self):
        if not self.thread:
            return

        self.thread.stop()
        if not self.thread.wait(1000):
            self.thread.terminate()
            self.thread.wait(500)

        self.thread.deleteLater()
        self.thread = None

    def start_recording(self):
        if self.is_running():
            self.thread.start_recording()

    def stop_recording(self):
        if self.is_running():
            self.thread.stop_recording()

    def current_roi(self):
        if not self.thread:
            return None
        return self.thread.requested_roi

    def current_size(self):
        if not self.thread:
            return None
        return self.thread.width, self.thread.height
