import h5py
import numpy as np
import pytest

from backend.event_processing import EVENT_CD_DTYPE
from backend.h5_source import H5SplitEventDataset, h5_event_dtype_names, h5_resolution, open_h5_events


def test_open_h5_events_finds_root_structured_dataset(tmp_path):
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("events", data=np.zeros(2, dtype=EVENT_CD_DTYPE))

    with h5py.File(path, "r") as h5_file:
        dataset = open_h5_events(h5_file)

        assert len(dataset) == 2
        assert h5_event_dtype_names(dataset) == EVENT_CD_DTYPE.names


def test_open_h5_events_finds_cd_events_dataset(tmp_path):
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("CD")
        group.create_dataset("events", data=np.zeros(3, dtype=EVENT_CD_DTYPE))

    with h5py.File(path, "r") as h5_file:
        dataset = open_h5_events(h5_file)

        assert len(dataset) == 3


def test_open_h5_events_supports_split_event_group(tmp_path):
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as h5_file:
        group = h5_file.create_group("events")
        group.create_dataset("x", data=np.array([1, 2], dtype=np.uint16))
        group.create_dataset("y", data=np.array([3, 4], dtype=np.uint16))
        group.create_dataset("t", data=np.array([100, 200], dtype=np.int64))
        group.create_dataset("p", data=np.array([1, 0], dtype=np.int8))

    with h5py.File(path, "r") as h5_file:
        dataset = open_h5_events(h5_file)
        events = dataset[:]

        assert isinstance(dataset, H5SplitEventDataset)
        assert events.dtype == EVENT_CD_DTYPE
        assert events.tolist() == [(1, 3, 1, 100), (2, 4, 0, 200)]


def test_open_h5_events_reports_unsupported_layout(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("other", data=np.zeros(1))

    with h5py.File(path, "r") as h5_file:
        with pytest.raises(ValueError, match="supported event dataset"):
            open_h5_events(h5_file)


def test_h5_resolution_uses_attrs_or_defaults(tmp_path):
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["width"] = 320
        h5_file.attrs["height"] = 240
        h5_file.create_dataset("events", data=np.zeros(1, dtype=EVENT_CD_DTYPE))

    with h5py.File(path, "r") as h5_file:
        assert h5_resolution(h5_file, h5_file["events"]) == (320, 240)
