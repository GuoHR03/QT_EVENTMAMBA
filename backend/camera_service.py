import logging
import queue

from backend.playback_config import PlaybackConfig, PlaybackConfigController, playback_restart_required
from backend.replay_clock import clamp_fraction
from backend.source_metadata_service import SourceMetadataService

LOGGER = logging.getLogger(__name__)

SOURCE_MODE_LIVE = "live"
SOURCE_MODE_FILE = "file"


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
        source_ready_callback=None,
    ):
        self.frame_queue = frame_queue
        self.image_signal = image_signal
        self.status_signal = status_signal
        self.finished_signal = finished_signal
        self.progress_signal = progress_signal
        self._thread_factory = thread_factory or _create_camera_thread
        self.metadata_service = metadata_service or SourceMetadataService()
        self._source_ready_callback = source_ready_callback
        self.thread = None
        self._thread_image_handler = None
        self._thread_status_handler = None
        self._thread_finished_handler = None
        self._thread_finished_token = None
        self._thread_progress_handler = None
        self._thread_source_ready_handler = None
        self.file_path = None
        self._source_mode = SOURCE_MODE_LIVE
        self._source_generation = 0
        self._source_ready = False
        self._analysis_enabled = False
        self.config_controller = PlaybackConfigController(PlaybackConfig())
        self.last_seek_fraction = 0.0
        self._last_progress_current_us = 0
        self._last_progress_total_us = 0

    def is_running(self):
        return self.thread is not None and self.thread.isRunning()

    @property
    def source_mode(self):
        return self._source_mode

    def is_recording(self):
        return (
            self.source_mode == SOURCE_MODE_LIVE
            and self.is_running()
            and bool(self.thread.is_recording)
        )

    def set_input_file(self, file_path, restart_if_running=False, config=None):
        if not file_path:
            return self._switch_source(
                None,
                restart_if_running=restart_if_running,
                config=config,
            )
        return self._switch_source(
            str(file_path),
            restart_if_running=restart_if_running,
            config=config,
        )

    def set_live_camera(self, config=None):
        return self._switch_source(
            None,
            restart_if_running=True,
            config=config,
        )

    def _switch_source(self, file_path, restart_if_running, config=None):
        was_running = self.is_running()
        if self.thread is not None:
            self.stop(emit_finished=False)

        self._source_generation += 1
        self.file_path = file_path
        self._source_mode = SOURCE_MODE_FILE if file_path else SOURCE_MODE_LIVE
        self._source_ready = False
        self.last_seek_fraction = 0.0
        self._last_progress_current_us = 0
        self._last_progress_total_us = 0
        self._clear_frame_queue()
        self.progress_signal.emit(0, 0)

        if was_running and restart_if_running:
            self.start(
                config=config or self.config_controller.get(),
                seek_fraction=0.0,
            )
            return True
        return False

    def start(
        self,
        config=None,
        seek_fraction=0.0,
        report_noise_filter_status=True,
    ):
        if self.thread is not None:
            self.stop(emit_finished=False)
        self._source_generation += 1
        config = config or self.config_controller.get()
        self.config_controller.set(config)
        self._source_ready = False
        self.last_seek_fraction = clamp_fraction(seek_fraction)

        kwargs = {
            "config_controller": self.config_controller,
            "target_queue": self.frame_queue,
            "seek_fraction": self.last_seek_fraction,
            "duration_hint_us": self._duration_hint_for_current_file(),
            "report_noise_filter_status": report_noise_filter_status,
            "analysis_enabled": self._analysis_enabled,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path

        self.thread = self._thread_factory(**kwargs)
        generation = self._source_generation
        thread = self.thread
        self._thread_image_handler = lambda image, timestamp: self._handle_image(
            thread,
            generation,
            image,
            timestamp,
        )
        self._thread_status_handler = lambda message: self._handle_status(
            thread,
            generation,
            message,
        )
        self.thread.image_signal.connect(self._thread_image_handler)
        self.thread.status_signal.connect(self._thread_status_handler)
        finished_token = {"emit": True}
        self._thread_finished_token = finished_token
        self._thread_finished_handler = lambda: self._handle_finished(
            thread,
            generation,
            finished_token,
        )
        self.thread.finished_signal.connect(self._thread_finished_handler)
        self._thread_progress_handler = (
            lambda current_us, total_us: self._handle_progress(
                thread,
                generation,
                current_us,
                total_us,
            )
        )
        self.thread.progress_signal.connect(self._thread_progress_handler)
        source_ready_signal = getattr(self.thread, "source_ready_signal", None)
        if source_ready_signal is not None:
            self._thread_source_ready_handler = (
                lambda width, height: self._handle_source_ready(
                    thread,
                    generation,
                    width,
                    height,
                )
            )
            source_ready_signal.connect(self._thread_source_ready_handler)
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
        seek_fraction = clamp_fraction(seek_fraction)
        config = self.config_controller.get()
        if self.thread is not None:
            self.stop(emit_finished=False)
        self._source_generation += 1
        self._source_ready = False
        self.last_seek_fraction = seek_fraction
        if self._last_progress_total_us > 0:
            self._last_progress_current_us = int(
                self._last_progress_total_us * seek_fraction
            )
        else:
            self._last_progress_current_us = 0
        self.progress_signal.emit(
            self._last_progress_current_us,
            self._last_progress_total_us,
        )
        self.start(
            config=config,
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

    def set_analysis_enabled(self, enabled):
        enabled = bool(enabled)
        changed = enabled != self._analysis_enabled
        self._analysis_enabled = enabled
        thread = self.thread
        setter = getattr(thread, "set_analysis_enabled", None)
        if callable(setter):
            setter(enabled)
        return changed

    def stop(self, emit_finished=True):
        if not self.thread:
            self._clear_frame_queue()
            return

        if bool(getattr(self.thread, "is_recording", False)):
            try:
                self.thread.stop_recording()
            except Exception:
                LOGGER.exception("Failed to stop RAW recording while stopping camera")

        if self._thread_image_handler is not None:
            try:
                self.thread.image_signal.disconnect(self._thread_image_handler)
            except (TypeError, RuntimeError, ValueError):
                pass
        if self._thread_status_handler is not None:
            try:
                self.thread.status_signal.disconnect(self._thread_status_handler)
            except (TypeError, RuntimeError, ValueError):
                pass

        if not emit_finished:
            if self._thread_finished_token is not None:
                self._thread_finished_token["emit"] = False
            try:
                self.thread.finished_signal.disconnect(self._thread_finished_handler)
            except (TypeError, RuntimeError, ValueError):
                pass
        if self._thread_progress_handler is not None:
            try:
                self.thread.progress_signal.disconnect(self._thread_progress_handler)
            except (TypeError, RuntimeError, ValueError):
                pass
        source_ready_signal = getattr(self.thread, "source_ready_signal", None)
        if source_ready_signal is not None and self._thread_source_ready_handler is not None:
            try:
                source_ready_signal.disconnect(self._thread_source_ready_handler)
            except (TypeError, RuntimeError, ValueError):
                pass

        self.thread.stop()
        if not self.thread.wait(2000):
            LOGGER.warning("Camera thread did not stop cooperatively within 2 seconds")
            self.thread.requestInterruption()
            if not self.thread.wait(1000):
                LOGGER.error("Camera thread still running; using forced termination as a last resort")
                self.thread.terminate()
                if not self.thread.wait(500):
                    raise RuntimeError(
                        "Camera thread could not be stopped; source switch aborted"
                    )

        self.thread.deleteLater()
        self.thread = None
        self._thread_image_handler = None
        self._thread_status_handler = None
        self._thread_finished_handler = None
        self._thread_finished_token = None
        self._thread_progress_handler = None
        self._thread_source_ready_handler = None
        self._source_ready = False
        self._clear_frame_queue()

    def start_recording(self):
        if self.source_mode != SOURCE_MODE_LIVE or not self.is_running():
            return False
        try:
            started = bool(self.thread.start_recording())
        except Exception:
            LOGGER.exception("Failed to start RAW recording")
            return False
        return started and bool(self.thread.is_recording)

    def stop_recording(self):
        if not self.is_running():
            return False
        try:
            stopped = bool(self.thread.stop_recording())
        except Exception:
            LOGGER.exception("Failed to stop RAW recording")
            return False
        return stopped and not bool(self.thread.is_recording)

    def current_config(self):
        return self.config_controller.get()

    def current_size(self):
        if not self.thread or not self._source_ready:
            return None
        return self.thread.width, self.thread.height

    def _duration_hint_for_current_file(self):
        return self.metadata_service.duration_hint(self.file_path)

    def _forward_progress(self, current_us, total_us):
        current_us = max(0, int(current_us or 0))
        total_us = max(0, int(total_us or 0))
        self._last_progress_current_us = current_us
        self._last_progress_total_us = total_us
        if total_us > 0 and self.file_path:
            self.metadata_service.record_duration(self.file_path, total_us)
        self.progress_signal.emit(current_us, total_us)

    def _handle_progress(self, thread, generation, current_us, total_us):
        if thread is not self.thread or generation != self._source_generation:
            return
        self._forward_progress(current_us, total_us)

    def _handle_image(self, thread, generation, image, timestamp):
        if thread is not self.thread or generation != self._source_generation:
            return
        self.image_signal.emit(image, timestamp)

    def _handle_status(self, thread, generation, message):
        if thread is not self.thread or generation != self._source_generation:
            return
        self.status_signal.emit(message)

    def _handle_finished(self, thread, generation, token):
        if not token.get("emit", False):
            return
        if generation != self._source_generation:
            return
        self.finished_signal.emit()

    def _ensure_raw_duration_scan(self):
        generation = self._source_generation
        self.metadata_service.ensure_duration_scan(
            self.file_path,
            callback=lambda input_path, duration_us: self._handle_duration_resolved(
                input_path,
                duration_us,
                generation=generation,
            ),
        )

    def _handle_duration_resolved(self, input_path, duration_us, generation=None):
        if self.file_path != input_path:
            return

        duration_us = max(0, int(duration_us or 0))
        if generation is not None and generation != self._source_generation:
            current_us = int(duration_us * self.last_seek_fraction)
        else:
            current_us = self._last_progress_current_us
        current_us = min(max(0, current_us), duration_us)
        self._last_progress_current_us = current_us
        self._last_progress_total_us = duration_us
        self.progress_signal.emit(current_us, duration_us)

    def _handle_source_ready(self, thread, generation, width, height):
        if thread is not self.thread or generation != self._source_generation:
            return
        self._source_ready = True
        if self._source_ready_callback is not None:
            self._source_ready_callback(int(width), int(height))

    def _clear_frame_queue(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except (AttributeError, queue.Empty):
                return


def _create_camera_thread(**kwargs):
    from backend.Camera import CameraThread

    return CameraThread(**kwargs)
