"""Coordinate camera transitions with the inference request generation."""

import logging
import queue

from backend.inference_lifecycle import STATE_RUNNING
from backend.model_contract import is_supported_model_mode


LOGGER = logging.getLogger(__name__)


class InferenceSessionCoordinator:
    """Own the ordering contract between camera state and inference traffic.

    Dependencies are supplied through getters because the application replaces
    services in tests and may replace them during future runtime reconfiguration.
    Keeping this object free of Qt types also makes the synchronization policy
    independently testable.
    """

    def __init__(
        self,
        camera_getter,
        inference_getter,
        payload_queue_getter,
        prediction_mode_getter,
    ):
        self._camera_getter = camera_getter
        self._inference_getter = inference_getter
        self._payload_queue_getter = payload_queue_getter
        self._prediction_mode_getter = prediction_mode_getter
        self.pending_network_thread = None

    @property
    def camera(self):
        return self._camera_getter()

    @property
    def inference(self):
        return self._inference_getter()

    @property
    def prediction_mode(self):
        return self._prediction_mode_getter()

    def enqueue_camera_config(self, width=None, height=None, network_thread=None):
        if not self.camera.is_running():
            return False
        if not is_supported_model_mode(self.prediction_mode):
            return False
        if width is None or height is None:
            camera_size = self.camera.current_size()
            if camera_size is None:
                return False
            width, height = camera_size
        payload = {
            "msg_type": "CONFIG",
            "width": int(width),
            "height": int(height),
            "prediction_mode": self.prediction_mode,
        }
        if network_thread is not None:
            replace_pending = getattr(network_thread, "replace_pending_payload", None)
            if callable(replace_pending):
                replace_pending(payload)
                return True
        return self.replace_queue_payload(payload)

    def replace_queue_payload(self, payload):
        payload_queue = self._payload_queue_getter()
        for _attempt in range(100):
            try:
                payload_queue.put_nowait(payload)
                return True
            except queue.Full:
                try:
                    payload_queue.get_nowait()
                except queue.Empty:
                    continue
        LOGGER.error("Could not replace pending camera payload with CONFIG")
        return False

    def active_network_thread(self):
        thread = getattr(self.inference, "network_thread", None)
        return thread if self.network_thread_is_usable(thread) else None

    def network_thread_is_usable(self, thread):
        current_thread = getattr(self.inference, "network_thread", None)
        if thread is None or thread is not current_thread:
            return False
        is_running = getattr(self.inference, "is_running", None)
        if callable(is_running):
            try:
                if not is_running():
                    return False
            except RuntimeError:
                return False
        elif getattr(self.inference, "state", STATE_RUNNING) != STATE_RUNNING:
            return False
        if not bool(getattr(thread, "running", True)):
            return False
        try:
            return bool(thread.isRunning())
        except (AttributeError, RuntimeError):
            return False

    def invalidate_for_camera_transition(self):
        self.camera.set_analysis_enabled(False)
        thread = self.active_network_thread()
        if thread is None:
            return None
        invalidate = getattr(thread, "invalidate_generation", None)
        if not callable(invalidate):
            return None
        invalidate()
        self.pending_network_thread = thread
        return thread

    def resume_network_thread(self, thread):
        if not self.network_thread_is_usable(thread):
            return False
        resume = getattr(thread, "resume_generation", None)
        if callable(resume):
            resume()
        self.camera.set_analysis_enabled(True)
        if self.pending_network_thread is thread:
            self.pending_network_thread = None
        return True

    def run_camera_transition(self, network_thread, operation):
        try:
            return operation()
        except Exception:
            self.resume_network_thread(network_thread)
            raise

    def handle_camera_source_ready(self, width, height):
        network_thread = self.pending_network_thread
        if network_thread is None:
            network_thread = self.active_network_thread()
        if network_thread is None:
            return False
        if self.enqueue_camera_config(width, height, network_thread=network_thread):
            return self.resume_network_thread(network_thread)
        return False

    def handle_camera_finished(self):
        return self.resume_network_thread(self.pending_network_thread)

    def configure_network_for_camera(self, network_thread=None):
        if network_thread is None:
            network_thread = self.invalidate_for_camera_transition()
        else:
            self.pending_network_thread = network_thread
        if network_thread is None:
            return False
        if not self.camera.is_running():
            self.resume_network_thread(network_thread)
            return False
        camera_size = self.camera.current_size()
        if camera_size is None:
            return False
        configured = self.enqueue_camera_config(
            *camera_size,
            network_thread=network_thread,
        )
        if configured or not is_supported_model_mode(self.prediction_mode):
            self.resume_network_thread(network_thread)
        return configured
