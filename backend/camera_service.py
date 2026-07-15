import logging

from backend.playback_config import PlaybackConfig, PlaybackConfigController, playback_restart_required
from backend.source_metadata_service import SourceMetadataService

LOGGER = logging.getLogger(__name__)


class CameraService:
    def __init__(
        self,
        frame_queue,
        image_signal,
        status_signal,
        finished_signal,
        progress_signal,
        thread_factory=None,
        metadata_service=None,
    ):
        self.frame_queue = frame_queue
        self.image_signal = image_signal
        self.status_signal = status_signal
        self.finished_signal = finished_signal
        self.progress_signal = progress_signal
        self._thread_factory = thread_factory or _create_camera_thread
        self.metadata_service = metadata_service or SourceMetadataService()
        self.thread = None
        self.file_path = None
        self.config_controller = PlaybackConfigController(PlaybackConfig())
        self.last_seek_fraction = 0.0
        self._last_progress_current_us = 0

    def is_running(self):
        return self.thread is not None and self.thread.isRunning()

    def is_recording(self):
        return self.is_running() and self.thread.is_recording

    def set_input_file(self, file_path):
        self.file_path = file_path
        self.last_seek_fraction = 0.0

    def start(
        self,
        config=None,
        seek_fraction=0.0,
        report_noise_filter_status=True,
    ):
        if self.thread is not None:
            self.stop(emit_finished=False)
        config = config or self.config_controller.get()
        self.config_controller.set(config)
        self.last_seek_fraction = _clamp_fraction(seek_fraction)

        kwargs = {
            "config_controller": self.config_controller,
            "target_queue": self.frame_queue,
            "seek_fraction": self.last_seek_fraction,
            "duration_hint_us": self._duration_hint_for_current_file(),
            "report_noise_filter_status": report_noise_filter_status,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path

        self.thread = self._thread_factory(**kwargs)
        self.thread.image_signal.connect(self.image_signal.emit)
        self.thread.status_signal.connect(self.status_signal.emit)
        self.thread.finished_signal.connect(self.finished_signal.emit)
        self.thread.progress_signal.connect(self._forward_progress)
        self.thread.start()
        self._ensure_raw_duration_scan()

    def restart(
        self,
        config=None,
        seek_fraction=None,
        report_noise_filter_status=True,
    ):
        config = config or self.config_controller.get()
        seek_fraction = self.last_seek_fraction if seek_fraction is None else seek_fraction
        self.stop(emit_finished=False)
        self.start(
            config=config,
            seek_fraction=seek_fraction,
            report_noise_filter_status=report_noise_filter_status,
        )

    def seek(self, seek_fraction):
        self.restart(
            config=self.config_controller.get(),
            seek_fraction=seek_fraction,
            report_noise_filter_status=False,
        )

    def apply_config(self, config):
        current = self.config_controller.get()
        if self.is_running() and playback_restart_required(current, config, self.file_path):
            self.restart(config=config)
            return True
        if self.is_running():
            self.thread.update_config(config)
        else:
            self.config_controller.set(config)
        return False

    def stop(self, emit_finished=True):
        if not self.thread:
            return

        if not emit_finished:
            try:
                self.thread.finished_signal.disconnect(self.finished_signal.emit)
            except (TypeError, RuntimeError):
                pass
        try:
            self.thread.progress_signal.disconnect(self._forward_progress)
        except (TypeError, RuntimeError):
            pass

        self.thread.stop()
        if not self.thread.wait(2000):
            LOGGER.warning("Camera thread did not stop cooperatively within 2 seconds")
            self.thread.requestInterruption()
            if not self.thread.wait(1000):
                LOGGER.error("Camera thread still running; using forced termination as a last resort")
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
        return self.config_controller.get().roi

    def current_config(self):
        return self.config_controller.get()

    def current_size(self):
        if not self.thread:
            return None
        return self.thread.width, self.thread.height

    def _duration_hint_for_current_file(self):
        return self.metadata_service.duration_hint(self.file_path)

    def _forward_progress(self, current_us, total_us):
        current_us = max(0, int(current_us or 0))
        total_us = max(0, int(total_us or 0))
        self._last_progress_current_us = current_us
        if total_us > 0 and self.file_path:
            self.metadata_service.record_duration(self.file_path, total_us)
        self.progress_signal.emit(current_us, total_us)

    def _ensure_raw_duration_scan(self):
        self.metadata_service.ensure_duration_scan(
            self.file_path,
            callback=self._handle_duration_resolved,
        )

    def _handle_duration_resolved(self, input_path, duration_us):
        if self.file_path != input_path:
            return

        current_us = min(max(0, self._last_progress_current_us), duration_us)
        self.progress_signal.emit(current_us, duration_us)


def _clamp_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, fraction))


def _create_camera_thread(**kwargs):
    from backend.Camera import CameraThread

    return CameraThread(**kwargs)
