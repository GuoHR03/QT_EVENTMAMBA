import numpy as np

from backend.event_processing import (
    filter_events_by_roi,
    replace_oldest_nowait,
    to_event_cd,
)


class EventWindowSlicer:
    """Split chronological CD events into fixed timestamp windows."""

    def __init__(self, interval_us):
        self.interval_us = max(1, int(interval_us))
        self.next_boundary_us = None
        self.buffer = None

    def consume(self, events):
        if events is None or len(events) == 0:
            return []

        events = np.ascontiguousarray(events)
        if self.next_boundary_us is None:
            self.next_boundary_us = int(events["t"][0]) + self.interval_us

        if self.buffer is not None and len(self.buffer) > 0:
            buffered = np.concatenate((self.buffer, events))
        else:
            buffered = events

        chunks = []
        while len(buffered) > 0 and int(buffered["t"][-1]) >= self.next_boundary_us:
            split_idx = int(np.searchsorted(buffered["t"], self.next_boundary_us, side="left"))
            if split_idx == 0:
                first_timestamp = int(buffered["t"][0])
                skipped = max(
                    1,
                    ((first_timestamp - self.next_boundary_us) // self.interval_us) + 1,
                )
                self.next_boundary_us += skipped * self.interval_us
                continue

            chunk = buffered[:split_idx]
            if len(chunk) > 0:
                chunks.append(np.ascontiguousarray(chunk))
            buffered = buffered[split_idx:]
            self.next_boundary_us += self.interval_us

        self.buffer = np.ascontiguousarray(buffered) if len(buffered) > 0 else None
        return chunks


class EventPipeline:
    """Normalize, filter, render, and fan out events for inference."""

    def __init__(
        self,
        roi_getter,
        noise_filter,
        renderer,
        inference_queue=None,
        inference_interval_us=None,
        analysis_enabled=None,
    ):
        self.roi_getter = roi_getter
        self.noise_filter = noise_filter
        self.renderer = renderer
        self.inference_queue = inference_queue
        self.analysis_enabled = analysis_enabled or (lambda: True)
        self.inference_slicer = (
            EventWindowSlicer(inference_interval_us)
            if inference_interval_us is not None
            else None
        )

    def process_events(self, events, render=True, infer=True):
        processed = self.prepare_events(events)
        if processed is None or len(processed) == 0:
            return processed

        if render:
            self.render_events(processed)
        if infer:
            self.enqueue_inference(processed)
        return processed

    def prepare_events(self, events):
        events = to_event_cd(events)
        if events is None or len(events) == 0:
            return np.empty(0, dtype=events.dtype) if events is not None else None

        events = filter_events_by_roi(events, self.roi_getter())
        events = self.noise_filter.apply(events)
        return np.ascontiguousarray(events)

    def render_events(self, events):
        if events is None or len(events) == 0 or self.renderer is None:
            return False
        self.renderer.process_events(np.ascontiguousarray(events))
        return True

    def enqueue_inference(self, events):
        if events is None or len(events) == 0:
            return 0

        if self.inference_slicer is None:
            chunks = [np.ascontiguousarray(events)]
        else:
            chunks = self.inference_slicer.consume(events)

        if self.inference_queue is None or not self.analysis_enabled():
            return 0

        queued = 0
        for chunk in chunks:
            if replace_oldest_nowait(self.inference_queue, chunk):
                queued += 1
        return queued
