import logging
import time
from threading import Lock

from PyQt6.QtCore import QThread

from backend.adaptive_inference_queue import AdaptiveInferenceQueueConsumer
from backend.inference_payload import InferencePayloadProcessor
from backend.inference_worker_control import INFERENCE_STOP_SIGNAL, enqueue_inference_stop


LOGGER = logging.getLogger(__name__)


class InferencePayloadWorker(QThread):
    """Build inference payloads from pre-sliced event windows."""

    def __init__(
        self,
        nn_queue,
        width,
        height,
        target_queue,
        analysis_enabled,
        roi=None,
        roi_getter=None,
        payload_publisher=None,
    ):
        super().__init__()
        self.nn_queue = nn_queue
        self._stop_signal_enqueued = False
        self._stop_lock = Lock()
        self._queue_consumer = AdaptiveInferenceQueueConsumer(
            nn_queue,
            INFERENCE_STOP_SIGNAL,
        )
        self._last_drop_report_at = 0.0
        self._reported_drops = 0
        self.processor = InferencePayloadProcessor(
            width=width,
            height=height,
            target_queue=target_queue,
            analysis_enabled=analysis_enabled,
            roi=roi,
            roi_getter=roi_getter,
            payload_publisher=payload_publisher,
        )

    def run(self):
        while True:
            events = self.nn_queue.get()
            events = self._queue_consumer.select(events)
            if events is INFERENCE_STOP_SIGNAL:
                break

            started_at = time.perf_counter()
            self.processor.process(events)
            self._queue_consumer.observe_processing(time.perf_counter() - started_at)
            self._report_adaptive_drops()

    def stop(self, discard_pending=True):
        with self._stop_lock:
            if self._stop_signal_enqueued:
                return
            self._stop_signal_enqueued = True
            enqueue_inference_stop(self.nn_queue, discard_pending)

    def _report_adaptive_drops(self):
        dropped = self._queue_consumer.dropped_windows
        if dropped <= self._reported_drops:
            return
        now = time.perf_counter()
        if now - self._last_drop_report_at < 5.0:
            return
        LOGGER.info(
            "Adaptive inference queue dropped %s stale window(s); "
            "processing EWMA %.2f ms, arrival EWMA %.2f ms",
            dropped - self._reported_drops,
            _milliseconds(self._queue_consumer.processing_ewma_s),
            _milliseconds(self._queue_consumer.arrival_ewma_s),
        )
        self._reported_drops = dropped
        self._last_drop_report_at = now


def _milliseconds(value):
    return 0.0 if value is None else value * 1000.0
