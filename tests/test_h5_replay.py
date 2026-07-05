import numpy as np
import pytest

from backend.event_processing import EVENT_CD_DTYPE
from backend.h5_replay import (
    h5_events_to_event_cd,
    h5_frame_interval_us,
    h5_replay_sleep_s,
    run_h5_replay_loop,
    select_h5_event_fields,
    split_events_for_frame,
)


def test_select_h5_event_fields_prefers_short_names():
    assert select_h5_event_fields(("x", "y", "p", "t")) == ("t", "p")


def test_select_h5_event_fields_accepts_aliases():
    assert select_h5_event_fields(("x", "y", "polarity", "timestamp")) == ("timestamp", "polarity")
    assert select_h5_event_fields(("x", "y", "pol", "ts")) == ("ts", "pol")


def test_select_h5_event_fields_reports_missing_fields():
    with pytest.raises(ValueError, match="x, time"):
        select_h5_event_fields(("y", "p"))


def test_h5_events_to_event_cd_converts_dtype():
    raw = np.array(
        [(1, 2, 100, 1), (3, 4, 200, 0)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )

    converted = h5_events_to_event_cd(raw, "timestamp", "polarity")

    assert converted.dtype == EVENT_CD_DTYPE
    assert converted.tolist() == [(1, 2, 1, 100), (3, 4, 0, 200)]


def test_split_events_for_frame_without_boundary_keeps_batch():
    events = np.array([(1, 1, 1, 100), (2, 2, 1, 150)], dtype=EVENT_CD_DTYPE)

    frame_part, remainder, reached = split_events_for_frame(events, 200)

    assert reached is False
    assert frame_part.tolist() == events.tolist()
    assert len(remainder) == 0


def test_split_events_for_frame_at_boundary():
    events = np.array([(1, 1, 1, 100), (2, 2, 1, 250), (3, 3, 1, 300)], dtype=EVENT_CD_DTYPE)

    frame_part, remainder, reached = split_events_for_frame(events, 250)

    assert reached is True
    assert frame_part.tolist() == [(1, 1, 1, 100)]
    assert remainder.tolist() == [(2, 2, 1, 250), (3, 3, 1, 300)]


def test_h5_timing_helpers():
    assert h5_frame_interval_us(50) == 20000
    assert h5_frame_interval_us(0) == 33333
    assert h5_replay_sleep_s(
        next_frame_target_time=120000,
        start_sensor_time=100000,
        start_real_time=10.0,
        now=10.005,
    ) == pytest.approx(0.015)


def test_run_h5_replay_loop_emits_frames_and_sleeps():
    dataset = np.array(
        [
            (1, 1, 100000, 1),
            (2, 2, 110000, 1),
            (3, 3, 130000, 0),
        ],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )
    emitted = []
    sleeps = []
    current_time = [0.0]

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration

    run_h5_replay_loop(
        events_dataset=dataset,
        dtype_names=dataset.dtype.names,
        fps=50,
        is_running=lambda: True,
        handle_frame_events=lambda events: emitted.append(events.copy()),
        now=lambda: current_time[0],
        sleep=fake_sleep,
        step=2,
    )

    assert [frame.tolist() for frame in emitted] == [
        [(1, 1, 1, 100000), (2, 2, 1, 110000)],
        [(3, 3, 0, 130000)],
    ]
    assert sleeps == [pytest.approx(0.04), pytest.approx(0.02)]


def test_run_h5_replay_loop_applies_replay_factor():
    dataset = np.array(
        [
            (1, 1, 100000, 1),
            (2, 2, 110000, 1),
            (3, 3, 130000, 0),
        ],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )
    sleeps = []
    current_time = [0.0]

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration

    run_h5_replay_loop(
        events_dataset=dataset,
        dtype_names=dataset.dtype.names,
        fps=50,
        is_running=lambda: True,
        handle_frame_events=lambda events: None,
        now=lambda: current_time[0],
        sleep=fake_sleep,
        replay_factor=2.0,
        step=2,
    )

    assert sleeps == [pytest.approx(0.02), pytest.approx(0.01)]


def test_run_h5_replay_loop_updates_replay_factor_while_running():
    dataset = np.array(
        [
            (1, 1, 100000, 1),
            (2, 2, 110000, 1),
            (3, 3, 130000, 0),
        ],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )
    replay_factor = [1.0]
    sleeps = []
    current_time = [0.0]

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration
        replay_factor[0] = 2.0

    run_h5_replay_loop(
        events_dataset=dataset,
        dtype_names=dataset.dtype.names,
        fps=50,
        is_running=lambda: True,
        handle_frame_events=lambda events: None,
        now=lambda: current_time[0],
        sleep=fake_sleep,
        replay_factor_getter=lambda: replay_factor[0],
        step=2,
    )

    assert sleeps == [pytest.approx(0.04), pytest.approx(0.01)]


def test_run_h5_replay_loop_updates_fps_while_running():
    dataset = np.array(
        [
            (1, 1, 100000, 1),
            (2, 2, 110000, 1),
            (3, 3, 130000, 0),
        ],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )
    fps = [50]
    sleeps = []
    current_time = [0.0]

    def fake_sleep(duration):
        sleeps.append(duration)
        current_time[0] += duration
        fps[0] = 100

    run_h5_replay_loop(
        events_dataset=dataset,
        dtype_names=dataset.dtype.names,
        fps=50,
        is_running=lambda: True,
        handle_frame_events=lambda events: None,
        now=lambda: current_time[0],
        sleep=fake_sleep,
        fps_getter=lambda: fps[0],
        step=2,
    )

    assert sleeps == [pytest.approx(0.04), pytest.approx(0.01)]


def test_run_h5_replay_loop_honors_stop_callback():
    dataset = np.array(
        [(1, 1, 100, 1), (2, 2, 200, 1)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("timestamp", "<i8"), ("polarity", "i1")],
    )
    emitted = []

    run_h5_replay_loop(
        events_dataset=dataset,
        dtype_names=dataset.dtype.names,
        fps=50,
        is_running=lambda: False,
        handle_frame_events=lambda events: emitted.append(events.copy()),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert emitted == []
