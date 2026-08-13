import queue

import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.event_pipeline import InferenceWindow
from backend.inference_payload import InferencePayloadProcessor


def test_inference_payload_processor_builds_one_payload_per_pre_sliced_window():
    events = np.zeros(1024, dtype=EVENT_CD_DTYPE)
    events["x"] = 100
    events["y"] = 100
    events["p"] = 1
    events["t"] = np.linspace(0, 19999, 1024, dtype=np.int64)
    target_queue = queue.Queue(maxsize=1)
    processor = InferencePayloadProcessor(
        width=640,
        height=480,
        target_queue=target_queue,
        analysis_enabled=lambda: True,
    )

    payload = processor.process(events)

    assert target_queue.get_nowait() is payload
    assert payload["msg_type"] == "EVENTS"
    assert payload["timestamp"] == 19999
    assert payload["data"].shape == (1024, 3)


def test_inference_payload_processor_reads_roi_dynamically():
    events = np.zeros(1024, dtype=EVENT_CD_DTYPE)
    events["x"] = 100
    events["y"] = 100
    events["p"] = 1
    events["t"] = np.arange(1024)
    roi = [(90, 90, 20, 20)]
    target_queue = queue.Queue(maxsize=2)
    processor = InferencePayloadProcessor(
        width=640,
        height=480,
        target_queue=target_queue,
        analysis_enabled=lambda: True,
        roi_getter=lambda: roi[0],
    )

    first = processor.process(events)
    roi[0] = (100, 100, 20, 20)
    second = processor.process(events)

    assert np.all(first["data"][:, 1] == 0.5)
    assert np.all(second["data"][:, 1] == 0.0)


def test_inference_window_uses_bound_roi_instead_of_new_runtime_roi():
    events = np.zeros(1024, dtype=EVENT_CD_DTYPE)
    events["x"] = 20
    events["y"] = 20
    events["p"] = 1
    events["t"] = np.arange(1024)
    published = []
    processor = InferencePayloadProcessor(
        width=640,
        height=480,
        target_queue=queue.Queue(),
        analysis_enabled=lambda: True,
        roi_getter=lambda: (15, 10, 20, 20),
        payload_publisher=lambda payload, generation: (
            published.append((payload, generation)) or True
        ),
    )

    payload = processor.process(
        InferenceWindow(events, roi=(10, 10, 20, 20), roi_generation=7)
    )

    assert payload is published[0][0]
    assert published[0][1] == 7
    assert np.all(payload["data"][:, 1] == 0.5)


def test_inference_payload_publisher_can_reject_stale_generation():
    events = np.zeros(1024, dtype=EVENT_CD_DTYPE)
    events["x"] = 10
    events["y"] = 10
    events["p"] = 1
    events["t"] = np.arange(1024)
    processor = InferencePayloadProcessor(
        width=640,
        height=480,
        target_queue=queue.Queue(),
        analysis_enabled=lambda: True,
        payload_publisher=lambda _payload, _generation: False,
    )

    assert processor.process(InferenceWindow(events, None, 0)) is None


def test_disabled_processor_returns_before_touching_event_data(monkeypatch):
    target_queue = queue.Queue()
    processor = InferencePayloadProcessor(
        width=640,
        height=480,
        target_queue=target_queue,
        analysis_enabled=lambda: False,
        roi_getter=lambda: (_ for _ in ()).throw(
            AssertionError("disabled processing must not read ROI")
        ),
    )
    monkeypatch.setattr(
        "backend.inference_payload.to_event_cd",
        lambda _events: (_ for _ in ()).throw(
            AssertionError("disabled processing must not convert events")
        ),
    )

    assert processor.process(object()) is None
    assert target_queue.empty()
