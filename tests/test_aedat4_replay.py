import numpy as np

from backend.aedat4_replay import (
    split_events_at_time,
)


EVENTS_DTYPE = [("x", "<u2"), ("y", "<u2"), ("polarity", "i1"), ("timestamp", "<i8")]


def test_split_events_at_time_returns_chunk_and_remainder():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 120), (3, 3, 1, 150)],
        dtype=EVENTS_DTYPE,
    )

    chunk, remaining, time_field = split_events_at_time(events, 130)

    assert time_field == "timestamp"
    assert chunk.tolist() == [(1, 1, 1, 100), (2, 2, 1, 120)]
    assert remaining.tolist() == [(3, 3, 1, 150)]


def test_split_events_at_time_handles_empty_first_chunk():
    events = np.array(
        [(1, 1, 1, 100), (2, 2, 1, 120)],
        dtype=EVENTS_DTYPE,
    )

    chunk, remaining, _ = split_events_at_time(events, 100)

    assert len(chunk) == 0
    assert remaining.tolist() == events.tolist()


def test_split_events_at_time_waits_until_target_is_reached():
    events = np.array([(1, 1, 1, 100)], dtype=EVENTS_DTYPE)

    chunk, remaining, time_field = split_events_at_time(events, 200)

    assert chunk is None
    assert remaining.tolist() == events.tolist()
    assert time_field == "timestamp"
