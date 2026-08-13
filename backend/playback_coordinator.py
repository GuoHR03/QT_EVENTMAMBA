import logging
import queue
from threading import Event, RLock

from backend.camera_source_runner import CameraRunContext
from backend.event_processing import (
    normalize_roi,
    put_latest,
    replace_oldest_nowait,
)
from backend.inference_worker_control import INFERENCE_STOP_SIGNAL
from backend.noise_filter import NoiseFilterPipeline
from backend.playback_config import PlaybackConfigController
from backend.replay_clock import clamp_fraction
from backend.playback_session import PlaybackSession
from backend.recording import RawRecorder
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH
from backend.source_factory import create_event_source


LOGGER = logging.getLogger(__name__)


class PlaybackCoordinator:
    """Assemble and coordinate one playback run independently of Qt signals."""

    def __init__(
        self,
        config_controller=None,
        target_queue=None,
        input_path="",
        seek_fraction=0.0,
        duration_hint_us=0,
        report_noise_filter_status=True,
        frame_callback=None,
        status_callback=None,
        progress_callback=None,
        source_ready_callback=None,
        analysis_enabled=False,
        source_factory=None,
        inference_worker_factory=None,
        session_factory=None,
        noise_filter_factory=None,
        recorder=None,
    ):
        self.config_controller = config_controller or PlaybackConfigController()
        self.target_queue = target_queue
        self.input_path = input_path
        self.seek_fraction = clamp_fraction(seek_fraction)
        self.duration_hint_us = max(0, int(duration_hint_us or 0))
        self.report_noise_filter_status = bool(report_noise_filter_status)

        self._frame_callback = frame_callback
        self._status_callback = status_callback
        self._progress_callback = progress_callback
        self._source_ready_callback = source_ready_callback
        self._source_factory = source_factory or create_event_source
        self._inference_worker_factory = (
            inference_worker_factory or _create_default_inference_worker
        )
        self._session_factory = session_factory or PlaybackSession
        self._noise_filter_factory = noise_filter_factory or NoiseFilterPipeline

        self._running = Event()
        self._running.set()
        self._config_update_lock = RLock()
        self._applied_config = self.config_controller.get()
        self._roi_generation = 0

        self.analysis_enabled = bool(analysis_enabled)
        self.width = DEFAULT_SENSOR_WIDTH
        self.height = DEFAULT_SENSOR_HEIGHT
        self.source_type = None
        self.device = None
        self.renderer = None
        self.source = None
        self.inference_worker = None
        self.session = None
        self.playback_start_time_us = 0
        self.playback_end_time_us = 0

        self.inference_queue = queue.Queue(maxsize=10)
        self.recorder = recorder if recorder is not None else RawRecorder()
        config = self._applied_config
        self.noise_filter = self._noise_filter_factory(
            config.noise_filter_type,
            config.noise_filter_threshold_us,
            status_callback=self._report_status,
            report_initial_status=self.report_noise_filter_status,
        )

    @property
    def is_running(self):
        return self._running.is_set()

    @property
    def is_recording(self):
        return self.recorder.is_recording

    def run(self):
        try:
            if not self.is_running:
                return
            self._initialize_source()
            if self.source is None:
                return

            config = self.config_controller.get()
            context = CameraRunContext(
                fps=config.fps,
                fps_getter=lambda: self.config_controller.get().fps,
                nn_interval_us=config.nn_interval_us,
                replay_factor=config.replay_factor,
                replay_factor_getter=lambda: self.config_controller.get().replay_factor,
                is_running=lambda: self.is_running,
                roi_getter=self.roi_tuple,
                roi_snapshot_getter=self.roi_snapshot,
                nn_queue=self.inference_queue,
                noise_filter=self.noise_filter,
                analysis_enabled=self.is_analysis_enabled,
                inference_publisher=self._publish_inference_window,
                inference_generation_is_current=(
                    self.is_inference_generation_current
                ),
                progress_callback=self._emit_playback_progress,
            )
            self.session = self._session_factory(
                source=self.source,
                context=context,
                inference_worker=self._create_inference_worker(),
            )
            if not self.is_running:
                self.session.stop()
            self.session.run()
        finally:
            if self.session is None and self.source is not None:
                self.source.close()
            self._running.clear()

    def stop(self):
        self._running.clear()
        if self.session is not None:
            self.session.stop()
            return
        if self.source is not None:
            request_stop = getattr(self.source, "request_stop", None)
            if callable(request_stop):
                request_stop()
        if self.inference_worker is not None:
            self.inference_worker.stop()

    def update_config(self, config):
        with self._config_update_lock:
            previous = self._applied_config
            roi_changed = previous.roi != config.roi
            self._apply_runtime_config(previous, config)
            self.config_controller.set(config)
            self._applied_config = config
            if roi_changed:
                self._roi_generation += 1
                self._discard_stale_roi_queues()
        if roi_changed:
            self._report_roi_update()
        if previous.replay_factor != config.replay_factor:
            LOGGER.info("Replay speed updated: %sx", config.replay_factor)
        return previous

    def is_analysis_enabled(self):
        with self._config_update_lock:
            return self.analysis_enabled

    def set_analysis_enabled(self, enabled):
        enabled = bool(enabled)
        with self._config_update_lock:
            if enabled == self.analysis_enabled:
                return False
            self.analysis_enabled = enabled
            # Reuse the window generation barrier: an item already dequeued by
            # the payload worker before disable must not become valid again if
            # inference is re-enabled quickly.
            self._roi_generation += 1
            self._discard_stale_roi_queues()
        LOGGER.info(
            "Inference event generation %s",
            "enabled" if enabled else "disabled",
        )
        return True

    def is_inference_generation_current(self, generation):
        with self._config_update_lock:
            return self.analysis_enabled and generation == self._roi_generation

    def start_recording(self):
        return self.recorder.start(self.device)

    def stop_recording(self):
        return self.recorder.stop(self.device)

    def roi_tuple(self):
        return self.roi_snapshot()[1]

    def roi_snapshot(self):
        with self._config_update_lock:
            roi = normalize_roi(
                self.config_controller.get().roi,
                self.width,
                self.height,
            )
            return self._roi_generation, roi

    def _initialize_source(self):
        config = self.config_controller.get()
        self.source = self._source_factory(
            input_path=self.input_path,
            fps=config.fps,
            palette_type=config.palette,
            frame_callback=self._on_source_frame,
            replay_factor=config.replay_factor,
            # The source queries the actual device geometry before clamping.
            # self.width/self.height still contain fallback defaults here.
            hardware_roi=config.roi,
            status_callback=self._report_status,
            replay_factor_getter=lambda: self.config_controller.get().replay_factor,
            seek_fraction=self.seek_fraction,
            duration_hint_us=self.duration_hint_us,
        )
        if self.source is None:
            return

        metadata = self.source.metadata()
        with self._config_update_lock:
            self.source_type = metadata.source_type
            self.device = self.source.device
            self.renderer = self.source.renderer
            self.width = metadata.width
            self.height = metadata.height
            latest_config = self.config_controller.get()
            self._apply_runtime_config(config, latest_config)
            self._applied_config = latest_config
        self._set_playback_range()
        if self._source_ready_callback is not None:
            self._source_ready_callback(self.width, self.height)

    def _create_inference_worker(self):
        self.inference_worker = self._inference_worker_factory(
            self.inference_queue,
            self.width,
            self.height,
            self.target_queue,
            self.is_analysis_enabled,
            roi_getter=self.roi_tuple,
            payload_publisher=self._publish_inference_payload,
        )
        return self.inference_worker

    def _publish_inference_payload(self, payload, roi_generation):
        with self._config_update_lock:
            if not self.analysis_enabled or (
                roi_generation is not None
                and roi_generation != self._roi_generation
            ):
                return False
            put_latest(self.target_queue, payload)
            return True

    def _publish_inference_window(self, window, roi_generation):
        with self._config_update_lock:
            if not self.analysis_enabled or (
                roi_generation is not None
                and roi_generation != self._roi_generation
            ):
                return False
            return replace_oldest_nowait(self.inference_queue, window)

    def _discard_stale_roi_queues(self):
        _discard_queue_items(
            self.inference_queue,
            preserve=lambda item: item is INFERENCE_STOP_SIGNAL,
        )
        _discard_queue_items(
            self.target_queue,
            preserve=lambda item: (
                isinstance(item, dict) and item.get("msg_type") == "CONFIG"
            ),
        )

    def _apply_runtime_config(self, previous, config):
        roi_changed = previous.roi != config.roi
        if self.renderer is not None and (
            previous.palette != config.palette or previous.fps != config.fps
        ):
            self.renderer.set_display_settings(config.palette, config.fps)
        if self.renderer is not None and roi_changed:
            reset_renderer = getattr(self.renderer, "reset", None)
            if callable(reset_renderer):
                reset_renderer()
        if (
            previous.noise_filter_type != config.noise_filter_type
            or previous.noise_filter_threshold_us != config.noise_filter_threshold_us
        ):
            self.noise_filter.update_settings(
                config.noise_filter_type,
                config.noise_filter_threshold_us,
            )
        elif roi_changed:
            self.noise_filter.reset()

    def _report_roi_update(self):
        roi = self.roi_tuple()
        if roi:
            x, y, width, height = roi
            LOGGER.info("ROI updated: x=%s, y=%s, w=%s, h=%s", x, y, width, height)
        else:
            LOGGER.info("ROI cleared")

    def _set_playback_range(self):
        metadata = self.source.metadata()
        self.playback_start_time_us = metadata.start_time_us
        self.playback_end_time_us = metadata.end_time_us
        seek_time_us = int(self.source.seek_time_us or self.playback_start_time_us)
        self._emit_playback_progress(seek_time_us)

    def _on_source_frame(self, timestamp_us, frame):
        if self._frame_callback is not None:
            self._frame_callback(frame.copy(), int(timestamp_us))

    def _report_status(self, message):
        LOGGER.info(message)
        if self._status_callback is not None:
            self._status_callback(message)

    def _emit_playback_progress(self, sensor_timestamp_us):
        if self._progress_callback is None:
            return
        total = self.playback_end_time_us - self.playback_start_time_us
        if total <= 0:
            self._progress_callback(max(0, int(sensor_timestamp_us or 0)), 0)
            return
        current = int(sensor_timestamp_us) - self.playback_start_time_us
        current = max(0, min(total, current))
        self._progress_callback(current, total)


def _create_default_inference_worker(*args, **kwargs):
    from backend.inference_payload_worker import InferencePayloadWorker

    return InferencePayloadWorker(*args, **kwargs)


def _discard_queue_items(target_queue, preserve):
    if target_queue is None or not hasattr(target_queue, "get_nowait"):
        return

    retained = []
    while True:
        try:
            item = target_queue.get_nowait()
        except queue.Empty:
            break
        if preserve(item):
            retained.append(item)

    for item in retained:
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            break
