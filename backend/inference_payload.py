import logging
import time

from backend.event_processing import build_inference_payload, put_latest, to_event_cd
from backend.event_pipeline import InferenceWindow
from backend.protocol import LOCAL_ROI_CONTEXT, LOCAL_TIMING_CONTEXT

LOGGER = logging.getLogger(__name__)


class InferencePayloadProcessor:
    """Convert one pre-sliced event window into one inference payload."""

    def __init__(
        self,
        width,
        height,
        target_queue,
        analysis_enabled,
        roi=None,
        roi_getter=None,
        payload_publisher=None,
    ):
        self.width = width
        self.height = height
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled
        self.roi = roi
        self.roi_getter = roi_getter
        self.payload_publisher = payload_publisher

    def process(self, events):
        process_started_at = time.perf_counter()
        if not self.analysis_enabled():
            return None

        roi_generation = None
        window_ready_at = process_started_at
        if isinstance(events, InferenceWindow):
            roi_generation = events.roi_generation
            roi = events.roi
            if events.ready_at > 0.0:
                window_ready_at = events.ready_at
            events = events.events
        else:
            roi = self.roi_getter() if self.roi_getter is not None else self.roi
        try:
            events = to_event_cd(events)
        except ValueError as exc:
            LOGGER.exception("Inference payload received invalid event data: %s", exc)
            return None

        if events is None or len(events) == 0 or self.target_queue is None:
            return None

        try:
            payload = build_inference_payload(
                events,
                width=self.width,
                height=self.height,
                roi=roi,
                fallback_normalization="crop",
            )
            if payload is not None:
                payload[LOCAL_ROI_CONTEXT] = roi
                payload[LOCAL_TIMING_CONTEXT] = {
                    "window_ready_at": window_ready_at,
                    "payload_started_at": process_started_at,
                    "payload_ready_at": time.perf_counter(),
                }
                if self.payload_publisher is not None:
                    if not self.payload_publisher(payload, roi_generation):
                        return None
                else:
                    put_latest(self.target_queue, payload)
            return payload
        except Exception as exc:
            LOGGER.exception("Inference payload error: %s", exc)
            return None
