from threading import Lock

from PyQt6.QtCore import QThread, pyqtSignal

from backend.inference_payload import InferencePayloadProcessor
from backend.inference_worker_control import INFERENCE_STOP_SIGNAL, enqueue_inference_stop


class InferencePayloadWorker(QThread):
    """Build inference payloads from pre-sliced event windows."""

    finished_signal = pyqtSignal()

    def __init__(self, nn_queue, width, height, target_queue, analysis_enabled, roi=None, roi_getter=None):
        super().__init__()
        self.nn_queue = nn_queue
        self.is_running = True
        self._stop_signal_enqueued = False
        self._stop_lock = Lock()
        self.processor = InferencePayloadProcessor(
            width=width,
            height=height,
            target_queue=target_queue,
            analysis_enabled=analysis_enabled,
            roi=roi,
            roi_getter=roi_getter,
        )

    def run(self):
        while True:
            events = self.nn_queue.get()
            if events is INFERENCE_STOP_SIGNAL:
                break

            self.processor.process(events)

        self.is_running = False
        self.finished_signal.emit()

    def stop(self, discard_pending=True):
        with self._stop_lock:
            if self._stop_signal_enqueued:
                return
            self._stop_signal_enqueued = True
            self.is_running = False
            enqueue_inference_stop(self.nn_queue, discard_pending)
