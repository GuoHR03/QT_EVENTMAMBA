import queue

import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.metavision_source import run_metavision_event_loop


class PassthroughFilter:
    def apply(self, events):
        return events


class FrameGenerator:
    def __init__(self):
        self.frames = []

    def process_events(self, events):
        self.frames.append(events)


def test_run_metavision_event_loop_filters_roi_and_replaces_queue():
    events = np.array(
        [(1, 1, 1, 100), (10, 10, 1, 110), (11, 11, 0, 120)],
        dtype=EVENT_CD_DTYPE,
    )
    target_queue = queue.Queue(maxsize=1)
    target_queue.put_nowait("old")
    frame_generator = FrameGenerator()

    run_metavision_event_loop(
        iterator=[events],
        is_running=lambda: True,
        roi_getter=lambda: (10, 10, 5, 5),
        noise_filter=PassthroughFilter(),
        frame_generator=frame_generator,
        nn_queue=target_queue,
    )

    queued = target_queue.get_nowait()
    assert queued.tolist() == [(10, 10, 1, 110), (11, 11, 0, 120)]
    assert frame_generator.frames[0].tolist() == queued.tolist()
