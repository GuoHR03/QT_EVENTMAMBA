import logging

from backend.event_processing import build_inference_payload, filter_events_by_roi, put_latest

LOGGER = logging.getLogger(__name__)


class H5FrameProcessor:
    """Process one H5 replay frame for display and optional inference."""

    def __init__(
        self,
        width,
        height,
        roi_getter,
        noise_filter,
        frame_generator,
        target_queue,
        analysis_enabled,
    ):
        self.width = width
        self.height = height
        self.roi_getter = roi_getter
        self.noise_filter = noise_filter
        self.frame_generator = frame_generator
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled

    def handle_frame_events(self, frame_events):
        roi = self.roi_getter()
        frame_events = filter_events_by_roi(frame_events, roi)
        frame_events = self.noise_filter.apply(frame_events)
        if len(frame_events) == 0:
            return False

        self.frame_generator.process_events(frame_events)
        self._enqueue_inference_payload(frame_events, roi)
        return True

    def _enqueue_inference_payload(self, frame_events, roi):
        if not self.analysis_enabled() or self.target_queue is None:
            return

        try:
            payload = build_inference_payload(
                frame_events,
                width=self.width,
                height=self.height,
                roi=roi,
                fallback_normalization="full",
            )
            if payload is not None:
                put_latest(self.target_queue, payload)
        except Exception as exc:
            LOGGER.exception("H5 inference enqueue error: %s", exc)
