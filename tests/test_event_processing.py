import queue

import numpy as np
import pytest

from backend.event_processing import (
    EVENT_CD_DTYPE,
    build_inference_payload,
    downsample_roi_normalize_events,
    event_time_field,
    filter_events_by_roi,
    normalize_noise_filter_type,
    normalize_roi,
    put_latest,
    replace_oldest_nowait,
    to_event_cd,
)


def test_normalize_noise_filter_type_accepts_aliases():
    assert normalize_noise_filter_type("off") == "none"
    assert normalize_noise_filter_type("Activity Noise") == "activity"
    assert normalize_noise_filter_type("spatio-temporal-contrast") == "stc"
    assert normalize_noise_filter_type("unknown") == "none"


def test_normalize_roi_clamps_to_sensor_bounds():
    assert normalize_roi((-10, 20, 50, 100), 640, 480) == (0, 20, 40, 100)
    assert normalize_roi((10, 10, 0, 100), 640, 480) is None
    assert normalize_roi(None, 640, 480) is None


def test_filter_events_by_roi_keeps_only_inside_events():
    events = np.array(
        [(5, 5, 1, 100), (12, 12, 0, 110), (20, 20, 1, 120)],
        dtype=EVENT_CD_DTYPE,
    )

    filtered = filter_events_by_roi(events, (10, 10, 8, 8))

    assert filtered.tolist() == [(12, 12, 0, 110)]


def test_to_event_cd_accepts_common_field_names():
    source = np.array(
        [(1, 2, 123, 1)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )

    converted = to_event_cd(source)

    assert converted.dtype == EVENT_CD_DTYPE
    assert converted.tolist() == [(1, 2, 1, 123)]
    assert event_time_field(source) == "timestamp"


def test_to_event_cd_rejects_missing_fields():
    source = np.array([(1, 2)], dtype=[("x", "<u2"), ("y", "<u2")])

    with pytest.raises(ValueError):
        to_event_cd(source)


def test_downsample_roi_normalize_events_returns_roi_relative_values():
    data = np.array(
        [
            [10, 20, 1000],
            [20, 30, 2000],
            [90, 90, 3000],
        ],
        dtype=np.float32,
    )

    x_norm, y_norm, t_norm = downsample_roi_normalize_events(data, (10, 20, 20, 20))

    assert np.allclose(x_norm, [0.0, 0.5])
    assert np.allclose(y_norm, [0.0, 0.5])
    assert len(t_norm) == 2
    assert t_norm[0] == pytest.approx(0.0)


def test_put_latest_preserves_config_when_queue_is_full():
    target = queue.Queue(maxsize=1)
    target.put_nowait({"msg_type": "CONFIG"})

    put_latest(target, {"msg_type": "EVENTS"})

    assert target.get_nowait() == {"msg_type": "CONFIG"}


def test_replace_oldest_nowait_puts_when_queue_has_space():
    target = queue.Queue(maxsize=2)

    assert replace_oldest_nowait(target, "new")
    assert target.get_nowait() == "new"


def test_replace_oldest_nowait_replaces_oldest_when_full():
    target = queue.Queue(maxsize=1)
    target.put_nowait("old")

    assert replace_oldest_nowait(target, "new")
    assert target.get_nowait() == "new"


def test_replace_oldest_nowait_handles_missing_queue():
    assert replace_oldest_nowait(None, "new") is False


def test_build_inference_payload_returns_none_when_too_few_points():
    events = np.array([(10, 10, 1, 100)], dtype=EVENT_CD_DTYPE)

    payload = build_inference_payload(events, width=640, height=480, target_points=2)

    assert payload is None


def test_build_inference_payload_uses_roi_relative_coordinates():
    events = np.array(
        [(10, 20, 1, 100), (20, 30, 1, 200)],
        dtype=EVENT_CD_DTYPE,
    )

    payload = build_inference_payload(
        events,
        width=640,
        height=480,
        roi=(10, 20, 20, 20),
        target_points=2,
    )

    assert payload["msg_type"] == "EVENTS"
    assert payload["cropped"] is True
    assert payload["timestamp"] == 200
    assert payload["data"].shape == (2, 3)
    assert np.allclose(payload["data"][:, 1], [0.0, 0.5])
    assert np.allclose(payload["data"][:, 2], [0.0, 0.5])


def test_build_inference_payload_can_use_full_frame_normalization():
    events = np.array(
        [(0, 0, 1, 100), (320, 240, 1, 200)],
        dtype=EVENT_CD_DTYPE,
    )

    payload = build_inference_payload(
        events,
        width=640,
        height=480,
        fallback_normalization="full",
        target_points=2,
    )

    assert payload["cropped"] is False
    assert np.allclose(payload["data"][:, 1], [0.0, 0.5])
    assert np.allclose(payload["data"][:, 2], [0.0, 0.5])
