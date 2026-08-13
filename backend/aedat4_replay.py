import numpy as np

from backend.event_processing import event_time_field


def split_events_at_time(buffer_events, boundary_time):
    time_field = event_time_field(buffer_events)
    if time_field is None:
        return None, buffer_events, None
    if len(buffer_events) == 0 or int(buffer_events[time_field][-1]) < boundary_time:
        return None, buffer_events, time_field

    split_idx = int(np.searchsorted(buffer_events[time_field], boundary_time))
    return buffer_events[:split_idx], buffer_events[split_idx:], time_field
