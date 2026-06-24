import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.settings import DEFAULT_FPS


def run_h5_replay_loop(
    events_dataset,
    dtype_names,
    fps,
    is_running,
    handle_frame_events,
    now,
    sleep,
    step=5000,
):
    total_events = len(events_dataset)
    current_idx = 0
    time_key, pol_key = select_h5_event_fields(dtype_names)
    frame_interval_us = h5_frame_interval_us(fps)
    start_real_time = now()
    start_sensor_time = None
    next_frame_target_time = None

    while is_running() and current_idx < total_events:
        events_for_this_frame = []

        while current_idx < total_events:
            end_idx = min(current_idx + step, total_events)
            raw_events = events_dataset[current_idx:end_idx]
            events = h5_events_to_event_cd(raw_events, time_key, pol_key)
            if len(events) == 0:
                current_idx = end_idx
                continue

            if start_sensor_time is None:
                start_sensor_time = int(events["t"][0])
                next_frame_target_time = start_sensor_time + frame_interval_us
                start_real_time = now()

            frame_part, _, reached_frame_boundary = split_events_for_frame(events, next_frame_target_time)
            if reached_frame_boundary:
                events_for_this_frame.append(frame_part)
                current_idx = current_idx + len(frame_part)
                break

            events_for_this_frame.append(frame_part)
            current_idx = end_idx

        if not events_for_this_frame:
            break

        frame_events = np.concatenate(events_for_this_frame)
        if len(frame_events) > 0:
            handle_frame_events(frame_events)

        next_frame_target_time += frame_interval_us
        sleep_time = h5_replay_sleep_s(
            next_frame_target_time,
            start_sensor_time,
            start_real_time,
            now(),
        )

        if sleep_time > 0.005:
            sleep(sleep_time)
        elif sleep_time < -0.2:
            start_real_time = now()
            start_sensor_time = next_frame_target_time - frame_interval_us


def select_h5_event_fields(dtype_names):
    names = set(dtype_names or ())
    time_key = _first_existing(names, ("t", "ts", "timestamp"))
    polarity_key = _first_existing(names, ("p", "pol", "polarity"))
    missing = []
    if "x" not in names:
        missing.append("x")
    if "y" not in names:
        missing.append("y")
    if time_key is None:
        missing.append("time")
    if polarity_key is None:
        missing.append("polarity")
    if missing:
        raise ValueError(f"H5 events dataset is missing fields: {', '.join(missing)}")
    return time_key, polarity_key


def h5_events_to_event_cd(raw_events, time_key, polarity_key):
    events = np.zeros(len(raw_events), dtype=EVENT_CD_DTYPE)
    events["x"] = raw_events["x"]
    events["y"] = raw_events["y"]
    events["p"] = raw_events[polarity_key]
    events["t"] = raw_events[time_key]
    return events


def split_events_for_frame(events, next_frame_target_time):
    if len(events) == 0:
        return events, events, False

    over_time_indices = np.where(events["t"] >= next_frame_target_time)[0]
    if len(over_time_indices) == 0:
        return events, events[:0], False

    split_idx = int(over_time_indices[0])
    return events[:split_idx], events[split_idx:], True


def h5_frame_interval_us(fps):
    fps = fps if fps > 0 else DEFAULT_FPS
    return int(1_000_000 / fps)


def h5_replay_sleep_s(next_frame_target_time, start_sensor_time, start_real_time, now):
    sensor_elapsed_s = (next_frame_target_time - start_sensor_time) / 1_000_000.0
    real_elapsed_s = now - start_real_time
    return sensor_elapsed_s - real_elapsed_s


def _first_existing(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None
