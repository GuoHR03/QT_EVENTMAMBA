import numpy as np
import pytest

from backend.aedat4_replay import (
    init_aedat4_timing,
    replay_sleep_s,
    should_emit_frame,
    should_reset_replay_clock,
    split_next_aedat4_nn_chunk,
)


EVENTS_DTYPE = [("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")]


def test_init_aedat4_timing_sets_frame_and_nn_targets():
    timing = init_aedat4_timing(
        first_timestamp=100000,
        frame_interval_us=33333,
        nn_interval_us=20000,
        now=12.5,
    )

    assert timing == {
        "start_sensor_time": 100000,
        "next_frame_time": 133333,
        "next_nn_time": 120000,
        "start_real_time": 12.5,
    }


def test_should_emit_frame_uses_available_time_field():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 150)],
        dtype=EVENTS_DTYPE,
    )

    assert should_emit_frame(events, 150)
    assert not should_emit_frame(events, 151)
    assert not should_emit_frame(np.array([], dtype=EVENTS_DTYPE), 1)


def test_split_next_aedat4_nn_chunk_returns_chunk_and_remainder():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 120), (3, 3, 1, 150)],
        dtype=EVENTS_DTYPE,
    )

    chunk, remaining, time_field = split_next_aedat4_nn_chunk(events, 130)

    assert time_field == "timestamp"
    assert chunk.tolist() == [(1, 1, 1, 100), (2, 2, 1, 120)]
    assert remaining.tolist() == [(3, 3, 1, 150)]


def test_split_next_aedat4_nn_chunk_handles_empty_first_chunk():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 120)],
        dtype=EVENTS_DTYPE,
    )

    chunk, remaining, _ = split_next_aedat4_nn_chunk(events, 100)

    assert len(chunk) == 0
    assert remaining.tolist() == events.tolist()


def test_split_next_aedat4_nn_chunk_waits_until_target_is_reached():
    events = np.array([(1, 1, 1, 100)], dtype=EVENTS_DTYPE)

    chunk, remaining, time_field = split_next_aedat4_nn_chunk(events, 200)

    assert chunk is None
    assert remaining.tolist() == events.tolist()
    assert time_field == "timestamp"


def test_replay_sleep_and_reset_helpers():
    assert replay_sleep_s(
        target_sensor_time=120000,
        start_sensor_time=100000,
        start_real_time=10.0,
        now=10.005,
    ) == pytest.approx(0.015)
    assert should_reset_replay_clock(-0.25)
    assert not should_reset_replay_clock(-0.1)
