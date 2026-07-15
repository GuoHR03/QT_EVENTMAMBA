from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH


SOURCE_AEDAT4 = "aedat4"
SOURCE_H5 = "h5"
SOURCE_METAVISION = "metavision"


def classify_input_source(input_path):
    path = (input_path or "").lower()
    if path.endswith(".aedat4"):
        return SOURCE_AEDAT4
    if path.endswith((".h5", ".hdf5")):
        return SOURCE_H5
    return SOURCE_METAVISION


def aedat4_resolution(reader):
    try:
        resolution = reader.getEventResolution()
        if isinstance(resolution, tuple):
            return int(resolution[0]), int(resolution[1])
        return int(resolution.width), int(resolution.height)
    except Exception:
        return DEFAULT_SENSOR_WIDTH, DEFAULT_SENSOR_HEIGHT


def aedat4_time_range(reader):
    candidates = (
        ("getTimeRange", ()),
        ("getTimeRangeUs", ()),
        ("getTimestampRange", ()),
    )
    for method_name, args in candidates:
        method = getattr(reader, method_name, None)
        if method is None:
            continue
        try:
            parsed = _parse_time_range(method(*args))
        except Exception:
            continue
        if parsed is not None:
            return parsed

    start_time = _call_first_int(reader, ("getStartTime", "getStartTimeUs", "getFirstTimestamp"))
    end_time = _call_first_int(reader, ("getEndTime", "getEndTimeUs", "getLastTimestamp"))
    duration = _call_first_int(reader, ("getDuration", "getDurationUs"))
    if start_time is not None and end_time is not None:
        return start_time, end_time
    if start_time is not None and duration is not None:
        return start_time, start_time + duration
    if duration is not None:
        return 0, duration
    return 0, 0


def _parse_time_range(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    start = _read_int_attr(value, ("start", "begin", "first", "min"))
    end = _read_int_attr(value, ("end", "last", "max"))
    if start is not None and end is not None:
        return start, end
    return None


def _call_first_int(source, names):
    for name in names:
        method = getattr(source, name, None)
        if method is None:
            continue
        try:
            return int(method())
        except Exception:
            continue
    return None


def _read_int_attr(source, names):
    for name in names:
        if not hasattr(source, name):
            continue
        value = getattr(source, name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            return int(value)
        except Exception:
            continue
    return None
