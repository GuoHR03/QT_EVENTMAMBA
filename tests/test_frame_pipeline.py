import queue

import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.frame_pipeline import H5FrameProcessor


class PassthroughFilter:
    def apply(self, events):
        return events


class EmptyFilter:
    def apply(self, events):
        return events[:0]


class FrameGenerator:
    def __init__(self):
        self.frames = []

    def process_events(self, events):
        self.frames.append(events.copy())


def test_h5_frame_processor_filters_roi_and_generates_frame():
    events = np.array(
        [(5, 5, 1, 100), (10, 10, 1, 200), (11, 11, 0, 300)],
        dtype=EVENT_CD_DTYPE,
    )
    frame_generator = FrameGenerator()

    processor = H5FrameProcessor(
        width=20,
        height=20,
        roi_getter=lambda: (10, 10, 5, 5),
        noise_filter=PassthroughFilter(),
        frame_generator=frame_generator,
        target_queue=None,
        analysis_enabled=lambda: True,
    )

    assert processor.handle_frame_events(events)
    assert frame_generator.frames[0].tolist() == [(10, 10, 1, 200), (11, 11, 0, 300)]


def test_h5_frame_processor_skips_empty_filtered_events():
    events = np.array([(10, 10, 1, 100)], dtype=EVENT_CD_DTYPE)
    frame_generator = FrameGenerator()

    processor = H5FrameProcessor(
        width=20,
        height=20,
        roi_getter=lambda: None,
        noise_filter=EmptyFilter(),
        frame_generator=frame_generator,
        target_queue=queue.Queue(maxsize=1),
        analysis_enabled=lambda: True,
    )

    assert not processor.handle_frame_events(events)
    assert frame_generator.frames == []


def test_h5_frame_processor_enqueues_full_frame_payload():
    events = np.zeros(1024, dtype=EVENT_CD_DTYPE)
    events["x"] = np.arange(1024) % 20
    events["y"] = np.arange(1024) % 20
    events["p"] = 1
    events["t"] = np.arange(1024) + 100
    target = queue.Queue(maxsize=1)
    frame_generator = FrameGenerator()

    processor = H5FrameProcessor(
        width=20,
        height=20,
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        frame_generator=frame_generator,
        target_queue=target,
        analysis_enabled=lambda: True,
    )

    assert processor.handle_frame_events(events)
    payload = target.get_nowait()
    assert payload["msg_type"] == "EVENTS"
    assert payload["cropped"] is False
    assert payload["timestamp"] == 1123
    assert payload["data"].shape == (1024, 3)
