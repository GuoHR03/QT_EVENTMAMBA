import logging

from backend.event_processing import build_inference_payload, put_latest, to_event_cd

LOGGER = logging.getLogger(__name__)


class InferencePayloadProcessor:
    """Convert one pre-sliced event window into one inference payload."""

    def __init__(self, width, height, target_queue, analysis_enabled, roi=None, roi_getter=None):
        self.width = width
        self.height = height
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled
        self.roi = roi
        self.roi_getter = roi_getter

    def process(self, events):
        try:
            events = to_event_cd(events)
        except ValueError as exc:
            LOGGER.exception("Inference payload received invalid event data: %s", exc)
            return None

        if events is None or len(events) == 0 or self.target_queue is None or not self.analysis_enabled():
            return None

        try:
            roi = self.roi_getter() if self.roi_getter is not None else self.roi
            payload = build_inference_payload(
                events,
                width=self.width,
                height=self.height,
                roi=roi,
                fallback_normalization="crop",
            )
            if payload is not None:
                put_latest(self.target_queue, payload)
            return payload
        except Exception as exc:
            LOGGER.exception("Inference payload error: %s", exc)
            return None
