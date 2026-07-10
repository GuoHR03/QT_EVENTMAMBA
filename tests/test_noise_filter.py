import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.noise_filter import NoiseFilterPipeline


class FakeOutput:
    def __init__(self):
        self.value = np.array([(2, 3, 1, 200)], dtype=EVENT_CD_DTYPE)

    def numpy(self, copy=False):
        if copy:
            return self.value.copy()
        return self.value


class FakeAlgorithm:
    def __init__(self):
        self.output = FakeOutput()
        self.received = None

    def get_empty_output_buffer(self):
        return self.output

    def process_events(self, events, output):
        self.received = events.copy()


def test_disabled_noise_filter_reports_and_passes_events_through():
    messages = []
    pipeline = NoiseFilterPipeline("none", status_callback=messages.append)
    events = np.array([(1, 2, 1, 100)], dtype=EVENT_CD_DTYPE)

    pipeline.initialize(640, 480)
    result = pipeline.apply(events)

    assert messages == ["[NoiseFilter] Disabled"]
    assert result is events
    assert not pipeline.enabled


def test_disabled_noise_filter_can_suppress_initial_status():
    messages = []
    pipeline = NoiseFilterPipeline("none", status_callback=messages.append, report_initial_status=False)

    pipeline.initialize(640, 480)

    assert messages == []


def test_noise_filter_apply_converts_common_event_fields():
    pipeline = NoiseFilterPipeline("activity")
    algorithm = FakeAlgorithm()
    pipeline.algorithm = algorithm
    pipeline.output = algorithm.get_empty_output_buffer()
    events = np.array(
        [(1, 2, 100, 1)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )

    result = pipeline.apply(events)

    assert algorithm.received.dtype == EVENT_CD_DTYPE
    assert algorithm.received.tolist() == [(1, 2, 1, 100)]
    assert result.tolist() == [(2, 3, 1, 200)]


def test_noise_filter_failure_reports_once_and_returns_raw_events():
    messages = []
    pipeline = NoiseFilterPipeline("activity", status_callback=messages.append)
    pipeline.algorithm = object()
    pipeline.output = FakeOutput()
    events = np.array([(1, 2, 1, 100)], dtype=EVENT_CD_DTYPE)

    assert pipeline.apply(events) is events
    assert pipeline.apply(events) is events

    assert len(messages) == 1
    assert messages[0].startswith("[NoiseFilter] Filtering failed")
