import pytest

from backend.replay_clock import ReplayClock, frame_interval_us, normalize_fps, replay_sleep_s, should_reset_replay_clock


def test_frame_interval_us_uses_default_for_invalid_fps():
    assert frame_interval_us(50) == 20000
    assert frame_interval_us(0) == 33333
    assert normalize_fps(None) == 30


def test_replay_clock_starts_and_advances_frames():
    clock = ReplayClock.start(first_sensor_time=100000, frame_interval_us=20000, now=10.0)

    assert clock.start_sensor_time == 100000
    assert clock.start_real_time == 10.0
    assert clock.next_frame_time == 120000

    clock.advance_frame()
    assert clock.next_frame_time == 140000


def test_replay_clock_sleep_until_sleeps_when_ahead():
    clock = ReplayClock.start(first_sensor_time=100000, frame_interval_us=20000, now=10.0)
    sleeps = []

    sleep_time = clock.sleep_until(120000, sleeps.append, now=lambda: 10.005)

    assert sleep_time == pytest.approx(0.015)
    assert sleeps == [pytest.approx(0.015)]


def test_replay_clock_sleep_until_applies_replay_factor():
    clock = ReplayClock.start(
        first_sensor_time=100000,
        frame_interval_us=20000,
        now=10.0,
        replay_factor=2.0,
    )
    sleeps = []

    sleep_time = clock.sleep_until(120000, sleeps.append, now=lambda: 10.004)

    assert sleep_time == pytest.approx(0.006)
    assert sleeps == [pytest.approx(0.006)]


def test_replay_clock_replay_factor_keeps_later_frames_on_speed_curve():
    clock = ReplayClock.start(
        first_sensor_time=100000,
        frame_interval_us=20000,
        now=0.0,
        replay_factor=2.0,
    )

    assert clock.sleep_time_s(140000, now=0.0) == pytest.approx(0.02)
    assert clock.sleep_time_s(160000, now=0.02) == pytest.approx(0.01)


def test_replay_clock_updates_frame_interval_for_next_frame():
    clock = ReplayClock.start(first_sensor_time=100000, frame_interval_us=20000, now=0.0)

    changed = clock.update_frame_interval_us(10000)
    clock.advance_frame()

    assert changed is True
    assert clock.frame_interval_us == 10000
    assert clock.next_frame_time == 130000


def test_replay_clock_reschedules_next_frame_from_anchor():
    clock = ReplayClock.start(first_sensor_time=100000, frame_interval_us=20000, now=0.0)

    clock.reschedule_next_frame(10000, anchor_sensor_time=100000)

    assert clock.frame_interval_us == 10000
    assert clock.next_frame_time == 110000


def test_replay_clock_updates_replay_factor_without_replaying_elapsed_time():
    clock = ReplayClock.start(
        first_sensor_time=100000,
        frame_interval_us=20000,
        now=0.0,
        replay_factor=1.0,
    )

    changed = clock.update_replay_factor(2.0, sensor_time=140000, now=0.04)

    assert changed is True
    assert clock.replay_factor == pytest.approx(2.0)
    assert clock.start_sensor_time == 140000
    assert clock.start_real_time == pytest.approx(0.04)
    assert clock.sleep_time_s(160000, now=0.04) == pytest.approx(0.01)


def test_replay_clock_sleep_until_uses_live_replay_factor_getter():
    replay_factor = [1.0]
    clock = ReplayClock.start(
        first_sensor_time=100000,
        frame_interval_us=20000,
        now=0.0,
        replay_factor=1.0,
    )
    sleeps = []

    replay_factor[0] = 2.0
    sleep_time = clock.sleep_until(
        160000,
        sleeps.append,
        now=lambda: 0.04,
        replay_factor_getter=lambda: replay_factor[0],
        factor_reset_sensor_time=140000,
    )

    assert sleep_time == pytest.approx(0.01)
    assert sleeps == [pytest.approx(0.01)]


def test_replay_clock_sleep_until_resets_when_lagging():
    clock = ReplayClock.start(first_sensor_time=100000, frame_interval_us=20000, now=10.0)

    sleep_time = clock.sleep_until(
        120000,
        lambda _: None,
        now=lambda: 10.5,
        reset_sensor_time=120000,
    )

    assert sleep_time == pytest.approx(-0.48)
    assert clock.start_sensor_time == 120000
    assert clock.start_real_time == 10.5


def test_replay_sleep_helpers():
    assert replay_sleep_s(
        target_sensor_time=120000,
        start_sensor_time=100000,
        start_real_time=10.0,
        now=10.005,
    ) == pytest.approx(0.015)
    assert should_reset_replay_clock(-0.25)
    assert not should_reset_replay_clock(-0.1)
