# Event stream capture, display, and inference fan-out.

import queue
import logging
import time
from PyQt6.QtCore import QThread, pyqtSignal
from backend.camera_source_runner import CameraRunContext, run_camera_source
from backend.camera_source_factory import (
    SOURCE_AEDAT4,
    SOURCE_H5,
    classify_input_source,
    create_aedat4_source,
    create_h5_source,
    create_metavision_source,
)
from backend.event_processing import (
    normalize_noise_filter_type as _normalize_noise_filter_type,
    normalize_roi as _normalize_roi,
)
from backend.inference_payload_worker import InferencePayloadWorker
from backend.noise_filter import NoiseFilterPipeline
from backend.recording import RawRecorder
from backend.replay_clock import normalize_fps
from backend.replay_speed import ReplaySpeedController, normalize_replay_factor
from backend.settings import (
    DEFAULT_FPS,
    DEFAULT_NN_INTERVAL_MS,
    DEFAULT_NOISE_FILTER_THRESHOLD_US,
    DEFAULT_REPLAY_FACTOR,
    DEFAULT_SENSOR_HEIGHT,
    DEFAULT_SENSOR_WIDTH,
)

LOGGER = logging.getLogger(__name__)


class CameraThread(QThread):
    """Read event streams and fan out frames plus inference batches."""
    finished_signal = pyqtSignal()

    def __init__(
        self,
        palette_type="Dark",
        fps=DEFAULT_FPS,
        nn_interval_ms=DEFAULT_NN_INTERVAL_MS,
        target_queue=None,
        file_path="",
        roi=None,
        replay_factor=DEFAULT_REPLAY_FACTOR,
        noise_filter_type="none",
        noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US,
    ):
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.recorder = RawRecorder()
        self.target_queue = target_queue
        self.analysis_enabled = True
        self.input_path = file_path
        self.palette_type = palette_type
        self.fps = normalize_fps(fps)
        self.nn_interval_us = int(nn_interval_ms * 1000)
        self.replay_factor = normalize_replay_factor(replay_factor)
        self.replay_speed_controller = ReplaySpeedController(self.replay_factor)
        self.width = DEFAULT_SENSOR_WIDTH
        self.height = DEFAULT_SENSOR_HEIGHT
        self.noise_filter_type = _normalize_noise_filter_type(noise_filter_type)
        self.noise_filter_threshold_us = max(1, int(noise_filter_threshold_us or DEFAULT_NOISE_FILTER_THRESHOLD_US))
        self.noise_filter = NoiseFilterPipeline(
            self.noise_filter_type,
            self.noise_filter_threshold_us,
            status_callback=self._report_status,
        )
        self._last_frame_emit_time = None
        self._frame_emit_interval_s = 1.0 / self.fps

        self.source_type = classify_input_source(self.input_path)
        self.is_aedat4 = self.source_type == SOURCE_AEDAT4
        self.is_h5 = self.source_type == SOURCE_H5

        self.nn_queue = queue.Queue(maxsize=10)

        self.nn_worker = None
        self.source = None

        self.requested_roi = roi
        self.roi = None
        self.roi_x = None
        self.roi_y = None
        self.roi_width = None
        self.roi_height = None

    def _on_cd_frame_cb(self, ts, frame):
        emit_time = time.perf_counter()
        if (
            self._last_frame_emit_time is not None
            and emit_time - self._last_frame_emit_time < self._frame_emit_interval_s
        ):
            return

        self._last_frame_emit_time = emit_time
        self.image_signal.emit(frame.copy(), int(ts))

    def _init_engine(self, palette_type):
        if self.is_aedat4:
            self.source = create_aedat4_source(
                self.input_path,
                palette_type,
                fps=self.fps,
                frame_callback=self._on_cd_frame_cb,
            )
            source = self.source
            self.device = source.device
            self.dv_reader = source.reader
            self.event_frame_gen = source.frame_generator
            self.width = source.width
            self.height = source.height
            self._set_roi(self.requested_roi)

        elif self.is_h5:
            self.source = create_h5_source(self.input_path, self.fps, palette_type, self._on_cd_frame_cb)
            source = self.source
            self.device = source.device
            self.h5_file = source.file
            self.events_dataset = source.events_dataset
            self.h5_dtypes = source.dtype_names
            self.event_frame_gen = source.frame_generator
            self.width = source.width
            self.height = source.height
            self._set_roi(self.requested_roi)

        else:
            self.device = None
            self._set_roi(self.requested_roi)
            self.source = create_metavision_source(
                input_path=self.input_path,
                delta_t_us=self.nn_interval_us,
                replay_factor=self.replay_factor,
                fps=self.fps,
                palette_type=palette_type,
                frame_callback=self._on_cd_frame_cb,
                hardware_roi=self._roi_tuple(),
                status_callback=self._report_status,
                replay_factor_getter=self.replay_speed_controller.get,
            )
            source = self.source
            if source is None:
                return
            self.device = source.device
            self.mv_iterator = source.iterator
            self.event_frame_gen = source.frame_generator
            self.width = source.width
            self.height = source.height
            self._set_roi(self.requested_roi)

    def _start_workers(self, palette_type):
        self.nn_worker = InferencePayloadWorker(
            self.nn_queue, self.nn_interval_us, self.width, self.height,
            self.target_queue, lambda: self.analysis_enabled, self._roi_tuple()
        )
        self.nn_worker.start()

    def _on_worker_finished(self):
        self.is_running = False

    def _set_roi(self, roi):
        normalized = _normalize_roi(roi, self.width, self.height)
        self.roi = normalized
        if normalized:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = normalized
        else:
            self.roi_x = None
            self.roi_y = None
            self.roi_width = None
            self.roi_height = None

    def _roi_tuple(self):
        if self.roi_x is None:
            return None
        return self.roi_x, self.roi_y, self.roi_width, self.roi_height

    def _report_status(self, message):
        LOGGER.info(message)
        self.status_signal.emit(message)

    def run(self):
        self._init_engine(self.palette_type)
        if self.source is None:
            self.is_running = False
            self.finished_signal.emit()
            return

        self.noise_filter.initialize(self.width, self.height)
        self._start_workers(self.palette_type)
        run_camera_source(
            self.source_type,
            self.source,
            CameraRunContext(
                fps=self.fps,
                fps_getter=lambda: self.fps,
                nn_interval_us=self.nn_interval_us,
                replay_factor=self.replay_factor,
                replay_factor_getter=self.replay_speed_controller.get,
                is_running=lambda: self.is_running,
                roi_getter=self._roi_tuple,
                image_callback=self.image_signal.emit,
                nn_queue=self.nn_queue,
                noise_filter=self.noise_filter,
                target_queue=self.target_queue,
                analysis_enabled=lambda: self.analysis_enabled,
            ),
        )

        self.is_running = False
        if self.nn_worker:
            self.nn_worker.is_running = False
            self.nn_worker.wait()

        self.finished_signal.emit()

    def stop(self):
        self.is_running = False

    def update_roi(self, roi):
        """Update the requested ROI; restart is required for hardware ROI."""
        self.requested_roi = roi
        self._set_roi(roi)
        if self._roi_tuple():
            LOGGER.info(
                "ROI updated: x=%s, y=%s, w=%s, h=%s",
                self.roi_x,
                self.roi_y,
                self.roi_width,
                self.roi_height,
            )
        else:
            LOGGER.info("ROI cleared")

    def set_replay_factor(self, replay_factor):
        self.replay_factor = self.replay_speed_controller.set(replay_factor)
        LOGGER.info("Replay speed updated: %sx", self.replay_factor)

    def set_display_settings(self, palette_type=None, fps=None):
        if palette_type is not None:
            self.palette_type = palette_type
        self.fps = normalize_fps(fps if fps is not None else self.fps)
        self._frame_emit_interval_s = 1.0 / self.fps
        self._last_frame_emit_time = None

        frame_generator = getattr(self, "event_frame_gen", None)
        if frame_generator is not None and hasattr(frame_generator, "set_display_settings"):
            frame_generator.set_display_settings(self.palette_type, self.fps)
        LOGGER.info("Display settings updated: palette=%s, fps=%s", self.palette_type, self.fps)

    def start_recording(self):
        self.is_recording = self.recorder.start(self.device)

    def stop_recording(self):
        self.recorder.stop(self.device)
        self.is_recording = self.recorder.is_recording

    image_signal = pyqtSignal(object, int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
