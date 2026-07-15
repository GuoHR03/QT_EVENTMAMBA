import queue

import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
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
