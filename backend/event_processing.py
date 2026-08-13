import queue

import numpy as np


EVENT_CD_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "i1"), ("t", "<i8")])

NOISE_FILTER_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "activity": "activity",
    "activity_noise": "activity",
    "activitynoisefilter": "activity",
    "trail": "trail",
    "trail_filter": "trail",
    "stc": "stc",
    "spatio_temporal_contrast": "stc",
    "spatiotemporalcontrast": "stc",
    "anti_flicker": "anti_flicker",
    "antiflicker": "anti_flicker",
    "flicker": "anti_flicker",
}

NOISE_FILTER_DISPLAY_NAMES = {
    "none": "None",
    "activity": "Activity",
    "trail": "Trail",
    "stc": "STC",
    "anti_flicker": "AntiFlicker",
}


def normalize_noise_filter_type(filter_type):
    key = str(filter_type or "none").strip().lower().replace("-", "_").replace(" ", "_")
    return NOISE_FILTER_ALIASES.get(key, "none")


def event_field_name(events, candidates):
    names = events.dtype.names or ()
    for name in candidates:
        if name in names:
            return name
    return None


def to_event_cd(events):
    if events is None:
        return None
    if len(events) == 0:
        return np.empty(0, dtype=EVENT_CD_DTYPE)

    names = events.dtype.names or ()
    time_field = event_field_name(events, ("t", "timestamp", "ts"))
    polarity_field = event_field_name(events, ("p", "pol", "polarity"))
    if "x" not in names or "y" not in names or time_field is None or polarity_field is None:
        raise ValueError("events must have x, y, polarity and timestamp fields")

    if events.dtype == EVENT_CD_DTYPE and time_field == "t" and polarity_field == "p":
        return np.ascontiguousarray(events)

    converted = np.empty(len(events), dtype=EVENT_CD_DTYPE)
    converted["x"] = events["x"]
    converted["y"] = events["y"]
    converted["p"] = events[polarity_field]
    converted["t"] = events[time_field]
    return converted


def event_time_field(events):
    return event_field_name(events, ("t", "timestamp", "ts"))


def normalize_roi(roi, src_width, src_height):
    if roi is None:
        return None

    try:
        x, y, width, height = [int(v) for v in roi]
        src_width = int(src_width)
        src_height = int(src_height)
    except (TypeError, ValueError, OverflowError):
        return None
    if width <= 0 or height <= 0 or src_width <= 0 or src_height <= 0:
        return None

    # Intersect the requested half-open rectangle with the sensor. An ROI
    # entirely outside the sensor is invalid; forcing it into a 1-pixel strip
    # would silently move the user's coordinates to an unrelated edge.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(src_width, x + width)
    y2 = min(src_height, y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def filter_events_by_roi(events, roi):
    if events is None or len(events) == 0 or not roi:
        return events

    x, y, width, height = roi
    mask = (
        (events["x"] >= x) & (events["x"] < x + width) &
        (events["y"] >= y) & (events["y"] < y + height)
    )
    return events[mask]


def empty_normalized_events():
    empty = np.array([], dtype=np.float32)
    return empty, empty, empty


def downsample_roi_normalize_events(data_numpy, roi):
    if data_numpy is None or len(data_numpy) == 0 or not roi:
        return empty_normalized_events()

    roi_x, roi_y, roi_width, roi_height = roi
    mask = (
        (data_numpy[:, 0] >= roi_x) & (data_numpy[:, 0] < roi_x + roi_width) &
        (data_numpy[:, 1] >= roi_y) & (data_numpy[:, 1] < roi_y + roi_height)
    )
    if not np.any(mask):
        return empty_normalized_events()

    cropped = data_numpy[mask]
    x_values = (cropped[:, 0] - roi_x) / roi_width
    y_values = (cropped[:, 1] - roi_y) / roi_height
    t_values = cropped[:, 2]
    x_values = np.clip(x_values, 0.0, 1.0)
    y_values = np.clip(y_values, 0.0, 1.0)

    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downsample_crop_normalize_events(data_numpy, src_width=640, src_height=480, dst_width=512, dst_height=512):
    if data_numpy is None or len(data_numpy) == 0:
        return empty_normalized_events()

    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    mask = (x_raw >= 96) & (x_raw < 608)
    if not np.any(mask):
        return empty_normalized_events()

    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]

    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downsample_normalize_events(data_numpy, src_width=640, src_height=480, dst_width=640, dst_height=480):
    if data_numpy is None or len(data_numpy) == 0:
        return empty_normalized_events()

    x_values = data_numpy[:, 0] * (dst_width / src_width)
    y_values = data_numpy[:, 1] * (dst_height / src_height)
    t_values = data_numpy[:, 2]
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)
    x_values = x_values / dst_width
    y_values = y_values / dst_height

    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def build_inference_payload(
    events,
    width,
    height,
    roi=None,
    fallback_normalization="crop",
    target_points=1024,
    timestamp=None,
    rng=None,
):
    if events is None or len(events) == 0:
        return None

    if timestamp is None:
        time_field = event_time_field(events)
        if time_field is None:
            return None
        timestamp = int(events[time_field][-1])

    time_field = event_time_field(events)
    if time_field is None:
        return None

    nn_events = np.column_stack((events["x"], events["y"], events[time_field]))
    if roi:
        x_norm, y_norm, t_norm = downsample_roi_normalize_events(
            nn_events,
            roi,
        )
        cropped = True
    elif fallback_normalization == "full":
        x_norm, y_norm, t_norm = downsample_normalize_events(
            nn_events,
            src_width=width,
            src_height=height,
        )
        cropped = False
    else:
        x_norm, y_norm, t_norm = downsample_crop_normalize_events(
            nn_events,
            src_width=width,
            src_height=height,
        )
        cropped = True

    if len(x_norm) < target_points:
        return None

    if len(x_norm) > target_points:
        random_choice = rng.choice if rng is not None else np.random.choice
        indices = random_choice(len(x_norm), target_points, replace=False)
        x_norm = x_norm[indices]
        y_norm = y_norm[indices]
        t_norm = t_norm[indices]

    clean_array = np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)
    return {
        "msg_type": "EVENTS",
        "data": clean_array,
        "timestamp": int(timestamp),
        "cropped": cropped,
    }


def put_latest(target_queue, payload):
    if target_queue is None:
        return

    try:
        if target_queue.full():
            existing = target_queue.get_nowait()
            if isinstance(existing, dict) and existing.get("msg_type") == "CONFIG":
                target_queue.put_nowait(existing)
                return
    except queue.Empty:
        pass
    except queue.Full:
        return

    try:
        target_queue.put_nowait(payload)
    except queue.Full:
        pass


def replace_oldest_nowait(target_queue, payload):
    if target_queue is None:
        return False

    try:
        target_queue.put_nowait(payload)
        return True
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
        target_queue.put_nowait(payload)
        return True
    except (queue.Empty, queue.Full):
        return False
