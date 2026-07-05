import numpy as np

from backend.event_processing import event_time_field
from backend.replay_clock import replay_sleep_s, should_reset_replay_clock


def init_aedat4_timing(first_timestamp, frame_interval_us, nn_interval_us, now):
    return {
        "start_sensor_time": int(first_timestamp),
        "next_frame_time": int(first_timestamp) + frame_interval_us,
        "next_nn_time": int(first_timestamp) + nn_interval_us,
        "start_real_time": now,
    }


def should_emit_frame(events, next_frame_time):
    if events is None or len(events) == 0:
        return False
    time_field = event_time_field(events)
    if time_field is None:
        return False
    return int(events[time_field][-1]) >= next_frame_time


def split_next_aedat4_nn_chunk(buffer_events, next_nn_time):
    time_field = event_time_field(buffer_events)
    if time_field is None:
        return None, buffer_events, None
    if len(buffer_events) == 0 or int(buffer_events[time_field][-1]) < next_nn_time:
        return None, buffer_events, time_field

    split_idx = int(np.searchsorted(buffer_events[time_field], next_nn_time))
    return buffer_events[:split_idx], buffer_events[split_idx:], time_field

