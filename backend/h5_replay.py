import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.h5_source import _first_existing, _h5_event_time_at
from backend.replay_clock import ReplayClock, frame_interval_us


def run_h5_replay_loop(
    events_dataset,
    dtype_names,
    fps,
    is_running,
    handle_frame_events,
    now,
    sleep,
    replay_factor=1.0,
    replay_factor_getter=None,
    fps_getter=None,
    start_time_us=0,
    progress_callback=None,
    step=5000,
):
    total_events = len(events_dataset)
    time_key, pol_key = select_h5_event_fields(dtype_names)
    current_idx = h5_find_start_index_for_time(events_dataset, time_key, start_time_us)
    frame_interval = _active_frame_interval_us(fps, fps_getter)
    clock = None

    while is_running() and current_idx < total_events:
        events_for_this_frame = []

        while current_idx < total_events:
            end_idx = min(current_idx + step, total_events)
            raw_events = events_dataset[current_idx:end_idx]
            events = h5_events_to_event_cd(raw_events, time_key, pol_key)
            if len(events) == 0:
                current_idx = end_idx
                continue

            if clock is None:
                active_replay_factor = replay_factor_getter() if replay_factor_getter is not None else replay_factor
                clock = ReplayClock.start(events["t"][0], frame_interval, now(), active_replay_factor)
            else:
                current_frame_start = clock.next_frame_time - clock.frame_interval_us
                clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), current_frame_start)

            frame_part, _, reached_frame_boundary = split_events_for_frame(events, clock.next_frame_time)
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
            if progress_callback is not None:
                progress_callback(int(frame_events["t"][-1]))

        current_frame_boundary = clock.next_frame_time
        clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), current_frame_boundary)
        clock.sleep_until(
            clock.next_frame_time,
            sleep,
            now,
            reset_sensor_time=current_frame_boundary,
            replay_factor_getter=replay_factor_getter,
            factor_reset_sensor_time=current_frame_boundary,
        )


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


def h5_find_start_index_for_time(events_dataset, time_key, start_time_us):
    total_events = len(events_dataset)
    if total_events <= 0 or start_time_us <= 0:
        return 0

    first_time = _h5_event_time_at(events_dataset, 0, time_key)
    if first_time is None or start_time_us <= first_time:
        return 0

    last_time = _h5_event_time_at(events_dataset, total_events - 1, time_key)
    if last_time is None:
        return 0
    if start_time_us > last_time:
        return total_events

    left = 0
    right = total_events
    while left < right:
        mid = (left + right) // 2
        timestamp = _h5_event_time_at(events_dataset, mid, time_key)
        if timestamp is None or timestamp < start_time_us:
            left = mid + 1
        else:
            right = mid
    return left


def split_events_for_frame(events, next_frame_target_time):
    if len(events) == 0:
        return events, events, False

    over_time_indices = np.where(events["t"] >= next_frame_target_time)[0]
    if len(over_time_indices) == 0:
        return events, events[:0], False

    split_idx = int(over_time_indices[0])
    return events[:split_idx], events[split_idx:], True


def _active_frame_interval_us(fps, fps_getter=None):
    return frame_interval_us(fps_getter() if fps_getter is not None else fps)
