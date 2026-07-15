from PyQt6.QtCore import QThread, pyqtSignal

from backend.playback_coordinator import PlaybackCoordinator


class CameraThread(QThread):
    """Qt thread and signal adapter for PlaybackCoordinator."""

    image_signal = pyqtSignal(object, int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int)

    def __init__(
        self,
        config_controller=None,
        target_queue=None,
        file_path="",
        seek_fraction=0.0,
        duration_hint_us=0,
        report_noise_filter_status=True,
    ):
        super().__init__()
        self.coordinator = PlaybackCoordinator(
            config_controller=config_controller,
            target_queue=target_queue,
            input_path=file_path,
            seek_fraction=seek_fraction,
            duration_hint_us=duration_hint_us,
            report_noise_filter_status=report_noise_filter_status,
            frame_callback=self.image_signal.emit,
            status_callback=self.status_signal.emit,
            progress_callback=self.progress_signal.emit,
        )

    def run(self):
        try:
            self.coordinator.run()
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.coordinator.stop()

    def update_config(self, config):
        return self.coordinator.update_config(config)

    def start_recording(self):
        return self.coordinator.start_recording()

    def stop_recording(self):
        return self.coordinator.stop_recording()

    @property
    def is_recording(self):
        return self.coordinator.is_recording

    @property
    def width(self):
        return self.coordinator.width

    @property
    def height(self):
        return self.coordinator.height
