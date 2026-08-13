from dataclasses import dataclass

import numpy as np

from backend.event_processing import (
    filter_events_by_roi,
    replace_oldest_nowait,
    to_event_cd,
)


_ROI_NOT_PROVIDED = object()


@dataclass(frozen=True)
class InferenceWindow:
    """One event window bound to the ROI generation that produced it."""

    events: object
    roi: object
    roi_generation: object


class EventWindowSlicer:
    """Split chronological CD events into fixed timestamp windows."""

    def __init__(self, interval_us):
        self.interval_us = max(1, int(interval_us))
        self.next_boundary_us = None
        self.buffer = None

    def reset(self):
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
        roi_snapshot_getter=None,
        inference_publisher=None,
        inference_generation_is_current=None,
    ):
        self.roi_getter = roi_getter
        self.noise_filter = noise_filter
        self.renderer = renderer
        self.inference_queue = inference_queue
        self.analysis_enabled = analysis_enabled or (lambda: True)
        self.roi_snapshot_getter = roi_snapshot_getter
        self.inference_publisher = inference_publisher
        self.inference_generation_is_current = inference_generation_is_current
        self._last_inference_roi_snapshot = _ROI_NOT_PROVIDED
        self._analysis_active = None
        self.inference_slicer = (
            EventWindowSlicer(inference_interval_us)
            if inference_interval_us is not None
            else None
        )

    def process_events(
        self,
        events,
        render=True,
        infer=True,
        roi_snapshot=_ROI_NOT_PROVIDED,
    ):
        if roi_snapshot is _ROI_NOT_PROVIDED:
            roi_generation, roi = self.current_roi_snapshot()
        else:
            roi_generation, roi = roi_snapshot
        processed = self.prepare_events(events, roi=roi)
        if processed is None or len(processed) == 0:
            return processed

        if render:
            self.render_events(processed)
        if infer:
            self.enqueue_inference(
                processed,
                roi=roi,
                roi_generation=roi_generation,
            )
        return processed

    def prepare_events(self, events, roi=_ROI_NOT_PROVIDED):
        events = to_event_cd(events)
        if events is None or len(events) == 0:
            return np.empty(0, dtype=events.dtype) if events is not None else None

        if roi is _ROI_NOT_PROVIDED:
            roi = self.roi_getter()
        events = filter_events_by_roi(events, roi)
        events = self.noise_filter.apply(events)
        return np.ascontiguousarray(events)

    def render_events(self, events):
        if events is None or len(events) == 0 or self.renderer is None:
            return False
        self.renderer.process_events(np.ascontiguousarray(events))
        return True

    def enqueue_inference(self, events, roi=None, roi_generation=None):
        if events is None or len(events) == 0:
            return 0
        if self.inference_queue is None:
            return 0
        enabled = bool(self.analysis_enabled())
        if not enabled:
            if (
                self._analysis_active is not False
                and self.inference_slicer is not None
            ):
                self.inference_slicer.reset()
            self._analysis_active = False
            self._last_inference_roi_snapshot = _ROI_NOT_PROVIDED
            return 0
        if (
            self.inference_generation_is_current is not None
            and not self.inference_generation_is_current(roi_generation)
        ):
            return 0

        roi_snapshot = (roi_generation, roi)
        reset_slicer = self._analysis_active is not True
        self._analysis_active = True
        if roi_snapshot != self._last_inference_roi_snapshot:
            reset_slicer = True
        if reset_slicer:
            if self.inference_slicer is not None:
                self.inference_slicer.reset()
        self._last_inference_roi_snapshot = roi_snapshot

        if self.inference_slicer is None:
            chunks = [np.ascontiguousarray(events)]
        else:
            chunks = self.inference_slicer.consume(events)

        queued = 0
        for chunk in chunks:
            payload = chunk
            if self.roi_snapshot_getter is not None:
                payload = InferenceWindow(
                    events=chunk,
                    roi=roi,
                    roi_generation=roi_generation,
                )
            if self.inference_publisher is not None:
                published = self.inference_publisher(payload, roi_generation)
            else:
                published = replace_oldest_nowait(self.inference_queue, payload)
            if published:
                queued += 1
        return queued

    def current_roi_snapshot(self):
        if self.roi_snapshot_getter is None:
            return None, self.roi_getter()
        roi_generation, roi = self.roi_snapshot_getter()
        return roi_generation, roi

    def is_roi_generation_current(self, roi_generation):
        if self.roi_snapshot_getter is None:
            return True
        current_generation, _roi = self.roi_snapshot_getter()
        return current_generation == roi_generation

    def is_roi_current(self, roi):
        return self.current_roi_snapshot()[1] == roi
