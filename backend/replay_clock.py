from dataclasses import dataclass

from backend.settings import DEFAULT_FPS
from backend.replay_speed import normalize_replay_factor


def normalize_fps(fps, default_fps=DEFAULT_FPS):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = default_fps
    return fps if fps > 0 else default_fps


def clamp_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, fraction))


def frame_interval_us(fps, default_fps=DEFAULT_FPS):
    fps = normalize_fps(fps, default_fps)
    return int(1_000_000 / fps)


def should_reset_replay_clock(sleep_time, threshold_s=-0.2):
    return sleep_time < threshold_s


@dataclass
class ReplayClock:
    frame_interval_us: int
    start_sensor_time: int
    start_real_time: float
    next_frame_time: int
    replay_factor: float = 1.0

    @classmethod
    def start(cls, first_sensor_time, frame_interval_us, now, replay_factor=1.0):
        start_sensor_time = int(first_sensor_time)
        return cls(
            frame_interval_us=frame_interval_us,
            start_sensor_time=start_sensor_time,
            start_real_time=now,
            next_frame_time=start_sensor_time + frame_interval_us,
            replay_factor=max(float(replay_factor or 1.0), 0.001),
        )

    def update_replay_factor(self, replay_factor, sensor_time, now):
        replay_factor = normalize_replay_factor(replay_factor)
        if replay_factor == self.replay_factor:
            return False
        self.replay_factor = replay_factor
        self.start_sensor_time = int(sensor_time)
        self.start_real_time = now
        return True

    def reschedule_next_frame(self, frame_interval_us, anchor_sensor_time):
        self.frame_interval_us = int(frame_interval_us)
        self.next_frame_time = int(anchor_sensor_time) + self.frame_interval_us

    def sleep_time_s(self, target_sensor_time, now):
        sensor_elapsed_s = (target_sensor_time - self.start_sensor_time) / 1_000_000.0
        real_elapsed_s = now - self.start_real_time
        return (sensor_elapsed_s / self.replay_factor) - real_elapsed_s

    def sleep_until(
        self,
        target_sensor_time,
        sleep,
        now,
        min_sleep_s=0.005,
        reset_threshold_s=-0.2,
        reset_sensor_time=None,
        replay_factor_getter=None,
        factor_reset_sensor_time=None,
    ):
        current_time = now()
        if replay_factor_getter is not None:
            factor_sensor_time = factor_reset_sensor_time
            if factor_sensor_time is None:
                factor_sensor_time = reset_sensor_time if reset_sensor_time is not None else target_sensor_time
            self.update_replay_factor(replay_factor_getter(), factor_sensor_time, current_time)

        sleep_time = self.sleep_time_s(target_sensor_time, current_time)
        if sleep_time > min_sleep_s:
            sleep(sleep_time)
            return sleep_time
        if should_reset_replay_clock(sleep_time, reset_threshold_s):
            self.start_real_time = current_time
            self.start_sensor_time = int(reset_sensor_time if reset_sensor_time is not None else target_sensor_time)
        return sleep_time

    def advance_frame(self):
        self.next_frame_time += self.frame_interval_us
