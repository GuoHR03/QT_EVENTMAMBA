import logging
import queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from backend.event_processing import build_inference_payload, put_latest, to_event_cd

LOGGER = logging.getLogger(__name__)


class InferencePayloadWorker(QThread):
    """Build inference payloads from event batches at a fixed interval."""

    finished_signal = pyqtSignal()

    def __init__(self, nn_queue, nn_interval_us, width, height, target_queue, analysis_enabled, roi=None):
        super().__init__()
        self.nn_queue = nn_queue
        self.nn_interval_us = nn_interval_us
        self.width = width
        self.height = height
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled
        self.roi = roi
        self.is_running = True

    def run(self):
        buffer = []
        next_nn_time = None

        while self.is_running:
            try:
                events = self.nn_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            try:
                events = to_event_cd(events)
            except ValueError as exc:
                LOGGER.exception("InferencePayloadWorker received invalid event data: %s", exc)
                continue

            buffer.append(events)

            if next_nn_time is None:
                next_nn_time = int(events["t"][-1]) + self.nn_interval_us
                continue

            if int(events["t"][-1]) >= next_nn_time:
                self._emit_payload(buffer)
                buffer = []
                next_nn_time += self.nn_interval_us

        self.finished_signal.emit()

    def _emit_payload(self, buffer):
        if not buffer or self.target_queue is None or not self.analysis_enabled():
            return

        try:
            nn_events = np.concatenate(buffer)
            payload = build_inference_payload(
                nn_events,
                width=self.width,
                height=self.height,
                roi=self.roi,
                fallback_normalization="crop",
            )
            if payload is not None:
                put_latest(self.target_queue, payload)
        except Exception as exc:
            LOGGER.exception("InferencePayloadWorker error: %s", exc)
