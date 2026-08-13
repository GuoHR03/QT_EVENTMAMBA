import queue

import numpy as np

from backend.event_pipeline import EventPipeline, EventWindowSlicer, InferenceWindow
from backend.event_processing import EVENT_CD_DTYPE


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


class TrackingSlicer:
    def __init__(self):
        self.consume_calls = 0
        self.reset_calls = 0

    def consume(self, _events):
        self.consume_calls += 1
        return []

    def reset(self):
        self.reset_calls += 1


def test_event_window_slicer_preserves_windows_across_input_batches():
    slicer = EventWindowSlicer(20000)
    first = np.array([(1, 1, 1, 0), (2, 2, 1, 10000)], dtype=EVENT_CD_DTYPE)
    second = np.array(
        [(3, 3, 0, 21000), (4, 4, 0, 30000), (5, 5, 1, 41000)],
        dtype=EVENT_CD_DTYPE,
    )

    assert slicer.consume(first) == []
    chunks = slicer.consume(second)

    assert [chunk["t"].tolist() for chunk in chunks] == [[0, 10000], [21000, 30000]]
    assert slicer.buffer["t"].tolist() == [41000]


def test_event_window_slicer_keeps_boundary_event_for_next_window():
    slicer = EventWindowSlicer(20000)
    events = np.array(
        [(1, 1, 1, 0), (2, 2, 1, 20000), (3, 3, 0, 40000)],
        dtype=EVENT_CD_DTYPE,
    )

    chunks = slicer.consume(events)

    assert [chunk["t"].tolist() for chunk in chunks] == [[0], [20000]]
    assert slicer.buffer["t"].tolist() == [40000]


def test_event_pipeline_normalizes_filters_renders_and_slices_inference():
    events = np.array(
        [(1, 1, 1, 0), (10, 10, 1, 10000), (11, 11, 0, 21000), (12, 12, 1, 31000)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")],
    )
    frames = FrameGenerator()
    inference_queue = queue.Queue(maxsize=4)
    pipeline = EventPipeline(
        roi_getter=lambda: (10, 10, 5, 5),
        noise_filter=PassthroughFilter(),
        renderer=frames,
        inference_queue=inference_queue,
        inference_interval_us=20000,
    )

    processed = pipeline.process_events(events)

    assert processed.dtype == EVENT_CD_DTYPE
    assert processed.tolist() == [(10, 10, 1, 10000), (11, 11, 0, 21000), (12, 12, 1, 31000)]
    assert frames.frames[0].tolist() == processed.tolist()
    assert inference_queue.get_nowait()["t"].tolist() == [10000, 21000]


def test_event_pipeline_can_defer_rendering_without_delaying_inference():
    events = np.array(
        [(1, 1, 1, 0), (2, 2, 0, 10000), (3, 3, 1, 21000)],
        dtype=EVENT_CD_DTYPE,
    )
    frames = FrameGenerator()
    inference_queue = queue.Queue()
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        renderer=frames,
        inference_queue=inference_queue,
        inference_interval_us=20000,
    )

    processed = pipeline.process_events(events, render=False)

    assert frames.frames == []
    assert inference_queue.get_nowait()["t"].tolist() == [0, 10000]
    assert pipeline.render_events(processed)
    assert frames.frames[0].tolist() == events.tolist()


def test_event_pipeline_skips_empty_filtered_events():
    frames = FrameGenerator()
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=EmptyFilter(),
        renderer=frames,
        inference_queue=queue.Queue(),
        inference_interval_us=20000,
    )

    processed = pipeline.process_events(np.array([(1, 1, 1, 0)], dtype=EVENT_CD_DTYPE))

    assert len(processed) == 0
    assert frames.frames == []


def test_event_pipeline_replaces_oldest_inference_chunk():
    inference_queue = queue.Queue(maxsize=1)
    inference_queue.put_nowait("old")
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        renderer=None,
        inference_queue=inference_queue,
    )
    events = np.array([(1, 1, 1, 0)], dtype=EVENT_CD_DTYPE)

    pipeline.process_events(events, render=False)

    assert inference_queue.get_nowait().tolist() == events.tolist()


def test_event_pipeline_reads_roi_dynamically_for_display_and_inference():
    roi = [None]
    frames = FrameGenerator()
    pipeline = EventPipeline(
        roi_getter=lambda: roi[0],
        noise_filter=PassthroughFilter(),
        renderer=frames,
    )
    events = np.array(
        [(1, 1, 1, 0), (10, 10, 0, 100)],
        dtype=EVENT_CD_DTYPE,
    )

    pipeline.process_events(events)
    roi[0] = (9, 9, 3, 3)
    pipeline.process_events(events)

    assert frames.frames[0].tolist() == events.tolist()
    assert frames.frames[1].tolist() == [(10, 10, 0, 100)]


def test_roi_generation_change_resets_slicer_and_binds_window_snapshot():
    snapshot = [0, None]
    inference_queue = queue.Queue()
    pipeline = EventPipeline(
        roi_getter=lambda: snapshot[1],
        roi_snapshot_getter=lambda: tuple(snapshot),
        noise_filter=PassthroughFilter(),
        renderer=None,
        inference_queue=inference_queue,
        inference_interval_us=20000,
    )
    first = np.array(
        [(1, 1, 1, 0), (2, 2, 1, 10000)],
        dtype=EVENT_CD_DTYPE,
    )
    second = np.array(
        [(10, 10, 1, 21000), (11, 11, 1, 30000), (12, 12, 1, 41000)],
        dtype=EVENT_CD_DTYPE,
    )

    pipeline.process_events(first, render=False)
    snapshot[:] = [1, (9, 9, 5, 5)]
    pipeline.process_events(second, render=False)

    window = inference_queue.get_nowait()
    assert isinstance(window, InferenceWindow)
    assert window.roi_generation == 1
    assert window.roi == (9, 9, 5, 5)
    assert window.events["t"].tolist() == [21000, 30000]


def test_disabled_inference_resets_without_consuming_or_enqueuing_events():
    enabled = [True]
    inference_queue = queue.Queue()
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        renderer=None,
        inference_queue=inference_queue,
        inference_interval_us=20000,
        analysis_enabled=lambda: enabled[0],
    )
    slicer = TrackingSlicer()
    pipeline.inference_slicer = slicer
    events = np.array(
        [(1, 1, 1, 0), (2, 2, 1, 25000)],
        dtype=EVENT_CD_DTYPE,
    )

    pipeline.process_events(events, render=False)
    reset_calls_while_enabled = slicer.reset_calls
    enabled[0] = False
    pipeline.process_events(events, render=False)

    assert slicer.consume_calls == 1
    assert slicer.reset_calls == reset_calls_while_enabled + 1
    assert inference_queue.empty()


def test_reenabled_inference_starts_a_fresh_window_after_disabled_events():
    enabled = [True]
    inference_queue = queue.Queue()
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        renderer=None,
        inference_queue=inference_queue,
        inference_interval_us=20000,
        analysis_enabled=lambda: enabled[0],
    )

    pipeline.process_events(
        np.array([(1, 1, 1, 0), (2, 2, 1, 10000)], dtype=EVENT_CD_DTYPE),
        render=False,
    )
    enabled[0] = False
    pipeline.process_events(
        np.array([(3, 3, 1, 21000), (4, 4, 1, 41000)], dtype=EVENT_CD_DTYPE),
        render=False,
    )
    assert inference_queue.empty()

    enabled[0] = True
    pipeline.process_events(
        np.array([(5, 5, 1, 100000), (6, 6, 1, 110000)], dtype=EVENT_CD_DTYPE),
        render=False,
    )
    assert inference_queue.empty()
    pipeline.process_events(
        np.array([(7, 7, 1, 121000)], dtype=EVENT_CD_DTYPE),
        render=False,
    )

    assert inference_queue.get_nowait()["t"].tolist() == [100000, 110000]
    assert inference_queue.empty()


def test_stale_inference_generation_is_rejected_before_slicing():
    inference_queue = queue.Queue()
    pipeline = EventPipeline(
        roi_getter=lambda: None,
        noise_filter=PassthroughFilter(),
        renderer=None,
        inference_queue=inference_queue,
        inference_interval_us=20000,
        analysis_enabled=lambda: True,
        inference_generation_is_current=lambda _generation: False,
    )
    slicer = TrackingSlicer()
    pipeline.inference_slicer = slicer
    events = np.array(
        [(1, 1, 1, 0), (2, 2, 1, 25000)],
        dtype=EVENT_CD_DTYPE,
    )

    pipeline.process_events(events, render=False)

    assert slicer.consume_calls == 0
    assert inference_queue.empty()
