import queue
import logging

from PyQt6.QtCore import QObject, pyqtSignal

from backend.camera_service import CameraService
from backend.inference_service import InferenceService
from backend.inference_session_coordinator import InferenceSessionCoordinator
from backend.model_contract import MODE_CENTER

LOGGER = logging.getLogger(__name__)


class BackendAPI(QObject):
    image_signal = pyqtSignal(object, int)
    camera_status_signal = pyqtSignal(str)
    prediction_signal = pyqtSignal(object, int)
    _network_result_signal = pyqtSignal(object, int, object)
    playback_finished_signal = pyqtSignal()
    playback_progress_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._network_result_signal.connect(self._handle_network_result)
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera = CameraService(
            self.camera_queue,
            self.image_signal,
            self.camera_status_signal,
            self.playback_finished_signal,
            self.playback_progress_signal,
            source_ready_callback=self._handle_camera_source_ready,
        )
        self.inference = InferenceService(
            self.camera_queue,
            prediction_callback=self._network_result_signal.emit,
        )
        self.prediction_mode = MODE_CENTER
        self._session_coordinator = InferenceSessionCoordinator(
            camera_getter=lambda: self.camera,
            inference_getter=lambda: self.inference,
            payload_queue_getter=lambda: self.camera_queue,
            prediction_mode_getter=lambda: self.prediction_mode,
        )
        self.playback_finished_signal.connect(self._handle_camera_finished)

    @property
    def _pending_camera_network_thread(self):
        """Compatibility alias for callers migrating to the coordinator."""
        return self._session_coordinator.pending_network_thread

    @_pending_camera_network_thread.setter
    def _pending_camera_network_thread(self, value):
        self._session_coordinator.pending_network_thread = value

    def is_camera_running(self):
        return self.camera.is_running()

    def is_inference_running(self):
        running = self.inference.is_running()
        if not running:
            self.camera.set_analysis_enabled(False)
        return running

    def is_recording(self):
        return self.camera.is_recording()

    @property
    def source_mode(self):
        return self.camera.source_mode

    def set_input_file(self, file_path, config=None, restart_if_running=False):
        network_thread = self._invalidate_network_for_camera_transition()
        restarted = self._run_camera_transition(
            network_thread,
            lambda: self.camera.set_input_file(
                file_path,
                config=config,
                restart_if_running=restart_if_running,
            ),
        )
        if not restarted:
            self._resume_network_thread(network_thread)
        return restarted

    def set_live_camera(self, config=None):
        network_thread = self._invalidate_network_for_camera_transition()
        restarted = self._run_camera_transition(
            network_thread,
            lambda: self.camera.set_live_camera(config=config),
        )
        if not restarted:
            self._resume_network_thread(network_thread)
        return restarted

    def start_camera(self, config):
        network_thread = self._invalidate_network_for_camera_transition()
        self._run_camera_transition(
            network_thread,
            lambda: self.camera.start(config=config),
        )

    def restart_camera(self, config=None):
        network_thread = self._invalidate_network_for_camera_transition()
        self._run_camera_transition(
            network_thread,
            lambda: self.camera.restart(config=config),
        )

    def seek_playback(self, seek_fraction):
        if not self.camera.is_running():
            return
        network_thread = self._invalidate_network_for_camera_transition()
        self._run_camera_transition(
            network_thread,
            lambda: self.camera.seek(seek_fraction),
        )

    def update_playback_config(self, config):
        network_thread = self._invalidate_network_for_camera_transition()
        restarted = self._run_camera_transition(
            network_thread,
            lambda: self.camera.apply_config(config),
        )
        if not restarted:
            # invalidate_generation() deliberately drops every pre-barrier
            # event and priority payload. Reinstall CONFIG before resuming so
            # an ROI/display hot update cannot let EVENTS overtake camera size.
            self._configure_network_for_camera(network_thread)
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
        network_thread = self._invalidate_network_for_camera_transition()
        try:
            return self.camera.stop(emit_finished=False)
        finally:
            self._resume_network_thread(network_thread)

    def set_prediction_mode(self, mode):
        mode_changed = self.prediction_mode != mode
        self.prediction_mode = mode
        LOGGER.info("Prediction mode set to: %s", mode)
        return bool(
            mode_changed
            and self.is_inference_running()
            and self.inference.weights_path
        )

    def start_recording(self):
        return self.camera.start_recording()

    def stop_recording(self):
        return self.camera.stop_recording()

    def start_eventmamba(self, weights_path, port=5555, host="127.0.0.1"):
        """Compatibility wrapper; UI code should use the two split phases."""
        self.start_eventmamba_backend(weights_path, port=port, host=host)
        try:
            return self.start_eventmamba_network(port=port, host=host)
        except Exception:
            self.stop_eventmamba_backend()
            raise

    def restart_eventmamba(self, port=5555, host="127.0.0.1"):
        """Compatibility wrapper; UI code should use the three split phases."""
        self.stop_eventmamba_network()
        try:
            self.restart_eventmamba_backend(port=port, host=host)
            return self.start_eventmamba_network(port=port, host=host)
        except Exception:
            self.stop_eventmamba_backend()
            raise

    def stop_eventmamba(self):
        errors = []
        try:
            self.stop_eventmamba_network()
        except Exception as exc:
            errors.append(exc)
        try:
            self.stop_eventmamba_backend()
        except Exception as exc:
            errors.append(exc)
        if errors:
            message = "; ".join(str(error) for error in errors)
            raise RuntimeError(f"Failed to stop inference service: {message}")
        return True

    def start_eventmamba_backend(
        self,
        weights_path,
        port=5555,
        host="127.0.0.1",
    ):
        # Model loading can take seconds.  Keep the camera's display path live,
        # but do not slice or normalize inference windows until the verified
        # network client is ready.
        self.camera.set_analysis_enabled(False)
        return self.inference.start_backend(
            weights_path,
            self.prediction_mode,
            port=port,
            host=host,
        )

    def start_eventmamba_network(self, port=5555, host="127.0.0.1"):
        # NetworkThread is a QObject and must be created by the UI thread.
        # Start it paused so CONFIG is always the first request for a source.
        self.camera.set_analysis_enabled(False)
        thread = self.inference.start_network(
            host=host,
            port=port,
            start_paused=True,
        )
        self._configure_network_for_camera(thread)
        return thread

    def stop_eventmamba_network(self):
        pending_thread = self._pending_camera_network_thread
        self.camera.set_analysis_enabled(False)
        network_thread = getattr(self.inference, "network_thread", None)
        invalidate = getattr(network_thread, "invalidate_generation", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:
                # The hard stop below still prevents further emits.  Keep
                # shutdown progressing even if the optional queue barrier
                # could not be installed.
                LOGGER.exception("Failed to invalidate inference generation")
        try:
            result = self.inference.stop_network()
        except Exception:
            # stop_network() retains a still-live QThread on failure. Keep
            # this reference as well so the paused generation remains
            # recoverable and a later stop retry can target the same thread.
            self._pending_camera_network_thread = pending_thread
            raise
        self._pending_camera_network_thread = None
        return result

    def stop_eventmamba_backend(self):
        self.camera.set_analysis_enabled(False)
        return self.inference.stop_backend()

    def restart_eventmamba_backend(self, port=5555, host="127.0.0.1"):
        self.camera.set_analysis_enabled(False)
        return self.inference.restart_backend(
            self.prediction_mode,
            port=port,
            host=host,
        )

    def cancel_eventmamba_start(self):
        # The UI owns the operation worker and can request cancellation in the
        # narrow interval before that worker enters start_backend/restart_backend.
        return self.inference.cancel_start(force=True)

    def close(self):
        self.stop_camera()
        self.stop_eventmamba()

    def _enqueue_camera_config(self, width=None, height=None, network_thread=None):
        return self._session_coordinator.enqueue_camera_config(
            width,
            height,
            network_thread,
        )

    def _replace_camera_queue_payload(self, payload):
        return self._session_coordinator.replace_queue_payload(payload)

    def _active_network_thread(self):
        return self._session_coordinator.active_network_thread()

    def _network_thread_is_usable(self, thread):
        return self._session_coordinator.network_thread_is_usable(thread)

    def _handle_network_result(self, result, timestamp, generation):
        """Forward only results belonging to the current UI-visible generation."""
        thread = getattr(self.inference, "network_thread", None)
        if thread is None:
            return
        is_current = getattr(thread, "is_generation_current", None)
        if is_current is None:
            return
        try:
            if not is_current(generation):
                return
        except RuntimeError:
            return
        self.prediction_signal.emit(result, int(timestamp))

    def _invalidate_network_for_camera_transition(self):
        return self._session_coordinator.invalidate_for_camera_transition()

    def _resume_network_thread(self, thread):
        return self._session_coordinator.resume_network_thread(thread)

    def _run_camera_transition(self, network_thread, operation):
        return self._session_coordinator.run_camera_transition(
            network_thread,
            operation,
        )

    def _handle_camera_source_ready(self, width, height):
        self._session_coordinator.handle_camera_source_ready(width, height)

    def _handle_camera_finished(self):
        # CameraService filters this signal by source generation. A delivered
        # finish therefore belongs to the current source, including an open
        # failure emitted just before its QThread returns.
        self._session_coordinator.handle_camera_finished()

    def _configure_network_for_camera(self, network_thread=None):
        return self._session_coordinator.configure_network_for_camera(
            network_thread,
        )
