import numpy as np

from backend.event_processing import EVENT_CD_DTYPE
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH


H5_EVENT_DATASET_CANDIDATES = (
    "events",
    "CD/events",
    "cd/events",
    "events/CD",
)

H5_EVENT_GROUP_CANDIDATES = (
    "events",
    "CD",
    "cd",
)


class H5SplitEventDataset:
    def __init__(self, group):
        self.group = group
        self.x_key = _first_existing(group, ("x",))
        self.y_key = _first_existing(group, ("y",))
        self.t_key = _first_existing(group, ("t", "ts", "timestamp"))
        self.p_key = _first_existing(group, ("p", "pol", "polarity"))
        missing = []
        if self.x_key is None:
            missing.append("x")
        if self.y_key is None:
            missing.append("y")
        if self.t_key is None:
            missing.append("time")
        if self.p_key is None:
            missing.append("polarity")
        if missing:
            raise ValueError(f"H5 event group is missing fields: {', '.join(missing)}")
        self.dtype = EVENT_CD_DTYPE

    def __len__(self):
        return len(self.group[self.x_key])

    def __getitem__(self, item):
        events = np.empty(len(self.group[self.x_key][item]), dtype=EVENT_CD_DTYPE)
        events["x"] = self.group[self.x_key][item]
        events["y"] = self.group[self.y_key][item]
        events["t"] = self.group[self.t_key][item]
        events["p"] = self.group[self.p_key][item]
        return events


def open_h5_events(h5_file):
    for path in H5_EVENT_DATASET_CANDIDATES:
        if path in h5_file and hasattr(h5_file[path], "dtype"):
            dataset = h5_file[path]
            if dataset.dtype.names:
                return dataset

    for path in H5_EVENT_GROUP_CANDIDATES:
        if path in h5_file and not hasattr(h5_file[path], "dtype"):
            try:
                return H5SplitEventDataset(h5_file[path])
            except ValueError:
                continue

    raise ValueError(
        "H5 file does not contain a supported event dataset. "
        "Expected structured dataset 'events' or 'CD/events', or split datasets x/y/t/p."
    )


def h5_event_dtype_names(events_dataset):
    return events_dataset.dtype.names or ()


def h5_resolution(h5_file, events_dataset=None):
    width = _attr_int(h5_file.attrs, ("width", "sensor_width"), None)
    height = _attr_int(h5_file.attrs, ("height", "sensor_height"), None)

    attrs = getattr(events_dataset, "attrs", None)
    if attrs is not None:
        width = width or _attr_int(attrs, ("width", "sensor_width"), None)
        height = height or _attr_int(attrs, ("height", "sensor_height"), None)

    return width or DEFAULT_SENSOR_WIDTH, height or DEFAULT_SENSOR_HEIGHT


def _first_existing(group, candidates):
    for candidate in candidates:
        if candidate in group:
            return candidate
    return None


def _attr_int(attrs, names, default):
    for name in names:
        if name in attrs:
            return int(attrs[name])
    return default
