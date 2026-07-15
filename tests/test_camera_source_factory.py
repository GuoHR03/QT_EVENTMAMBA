from types import SimpleNamespace

import numpy as np

from backend.event_source import Aedat4Source, SourceMetadata
from backend.renderer_factory import (
    MetavisionFrameRenderer,
    _create_periodic_frame_generator,
    _instantiate_periodic_frame_generator,
)
from backend.source_factory import (
    create_aedat4_source,
    create_event_source,
    create_h5_source,
    create_metavision_source,
)
from backend.source_metadata import (
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    classify_input_source,
)
from backend.event_processing import EVENT_CD_DTYPE


def test_classify_input_source_detects_aedat4():
    assert classify_input_source("sample.AEDAT4") == SOURCE_AEDAT4


def test_classify_input_source_detects_h5_variants():
    assert classify_input_source("sample.h5") == SOURCE_H5
    assert classify_input_source("sample.HDF5") == SOURCE_H5


def test_classify_input_source_defaults_to_metavision():
    assert classify_input_source("") == SOURCE_METAVISION
    assert classify_input_source("recording.raw") == SOURCE_METAVISION


def test_aedat4_source_records_renderer():
    source = Aedat4Source(
        reader=object(),
        renderer=object(),
        metadata=SourceMetadata(SOURCE_AEDAT4, 640, 480),
    )

    assert source.renderer is not None
    assert source.metadata().width == 640


def test_create_event_source_dispatches_h5_without_requiring_sidecar(monkeypatch):
    expected = object()
    calls = {}

    def fake_h5_source(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("backend.source_factory.create_h5_source", fake_h5_source)

    source = create_event_source(
        input_path="events.h5",
        fps=30,
        palette_type="Dark",
        frame_callback=None,
        seek_fraction=0.25,
    )

    assert source is expected
    assert calls["args"][:3] == ("events.h5", 30, "Dark")
    assert calls["kwargs"]["seek_fraction"] == 0.25


def test_h5_source_builds_metadata_from_event_dataset(tmp_path, monkeypatch):
    import h5py

    input_path = tmp_path / "events.h5"
    events = np.array(
        [(1, 2, 1, 100), (3, 4, 0, 200)],
        dtype=EVENT_CD_DTYPE,
    )
    with h5py.File(input_path, "w") as h5_file:
        h5_file.attrs["width"] = 320
        h5_file.attrs["height"] = 240
        h5_file.create_dataset("events", data=events)

    monkeypatch.setattr(
        "backend.source_factory.create_metavision_renderer",
        lambda *args: object(),
    )

    source = create_h5_source(str(input_path), 30, "Dark", None, seek_fraction=0.5)
    metadata = source.metadata()

    assert (metadata.width, metadata.height) == (320, 240)
    assert (metadata.start_time_us, metadata.end_time_us) == (100, 200)
    assert metadata.duration_us == 100
    assert source.seek_time_us == 150
    source.close()


def test_aedat4_source_builds_metadata_from_reader(monkeypatch):
    class FakeReader:
        def getEventResolution(self):
            return 346, 260

        def getTimeRange(self):
            return 1000, 101000

    reader = FakeReader()
    fake_dv = SimpleNamespace(
        io=SimpleNamespace(MonoCameraRecording=lambda _path: reader),
    )
    monkeypatch.setitem(__import__("sys").modules, "dv_processing", fake_dv)

    source = create_aedat4_source("events.aedat4", "Dark", seek_fraction=0.25)
    metadata = source.metadata()

    assert (metadata.width, metadata.height) == (346, 260)
    assert (metadata.start_time_us, metadata.end_time_us) == (1000, 101000)
    assert metadata.duration_us == 100000
    assert source.seek_time_us == 26000


def test_metavision_renderer_rebuilds_generator_on_display_update():
    created = []

    class FakeGenerator:
        def __init__(self, width, height, fps, palette_type, frame_callback):
            self.width = width
            self.height = height
            self.fps = fps
            self.palette_type = palette_type
            self.frame_callback = frame_callback
            self.events = []

        def process_events(self, events):
            self.events.append(events)

    def fake_factory(width, height, fps, palette_type, frame_callback):
        generator = FakeGenerator(width, height, fps, palette_type, frame_callback)
        created.append(generator)
        return generator

    frame_generator = MetavisionFrameRenderer(
        width=640,
        height=480,
        fps=30,
        palette_type="Dark",
        frame_callback=object(),
        generator_factory=fake_factory,
    )

    frame_generator.process_events("before")
    changed = frame_generator.set_display_settings("Light", 60)
    frame_generator.process_events("after")
    reset = frame_generator.reset()
    frame_generator.close()
    frame_generator.process_events("closed")

    assert changed is True
    assert reset is True
    assert len(created) == 3
    assert created[0].events == ["before"]
    assert created[1].palette_type == "Light"
    assert created[1].fps == 60
    assert created[1].events == ["after"]
    assert created[2].events == []


def test_periodic_frame_generator_uses_frame_interval_as_accumulation(monkeypatch):
    created = {}

    class FakeColorPalette:
        Dark = "dark"
        Light = "light"
        CoolWarm = "cool_warm"
        Gray = "gray"

    class FakePeriodicFrameGenerationAlgorithm:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "metavision_sdk_core",
        type(
            "FakeMetavisionSdkCore",
            (),
            {
                "PeriodicFrameGenerationAlgorithm": FakePeriodicFrameGenerationAlgorithm,
                "ColorPalette": FakeColorPalette,
            },
        ),
    )

    _create_periodic_frame_generator(640, 480, 30, "Dark", None)

    assert created["sensor_width"] == 640
    assert created["sensor_height"] == 480
    assert created["fps"] == 30
    assert created["accumulation_time_us"] == 33333
    assert created["palette"] == "dark"


def test_periodic_frame_generator_falls_back_to_accumulation_setter():
    class LegacyFrameGenerationAlgorithm:
        def __init__(self, **kwargs):
            if "accumulation_time_us" in kwargs:
                raise TypeError("unsupported keyword")
            self.kwargs = kwargs
            self.accumulation_time_us = None

        def set_accumulation_time_us(self, accumulation_time_us):
            self.accumulation_time_us = accumulation_time_us

    generator = _instantiate_periodic_frame_generator(
        LegacyFrameGenerationAlgorithm,
        width=640,
        height=480,
        fps=30,
        accumulation_time_us=33333,
        palette="dark",
    )

    assert generator.kwargs == {
        "sensor_width": 640,
        "sensor_height": 480,
        "fps": 30,
        "palette": "dark",
    }
    assert generator.accumulation_time_us == 33333


def test_metavision_source_uses_duration_hint_for_raw_seek(monkeypatch):
    calls = {}

    class FakeIterator:
        def get_size(self):
            return 480, 640

    def fake_iterator(input_path, device, delta_t_us, replay_factor, replay_factor_getter=None, start_ts=0):
        calls["input_path"] = input_path
        calls["start_ts"] = start_ts
        return FakeIterator()

    monkeypatch.setattr("backend.source_factory.create_metavision_iterator", fake_iterator)
    monkeypatch.setattr("backend.source_factory.create_metavision_renderer", lambda *args: object())

    source = create_metavision_source(
        input_path="recording.raw",
        delta_t_us=20000,
        replay_factor=1.0,
        fps=30,
        palette_type="Dark",
        frame_callback=None,
        seek_fraction=0.5,
        duration_hint_us=123456,
    )

    assert calls["input_path"] == "recording.raw"
    assert calls["start_ts"] == 60000
    assert source.end_time_us == 123456
    assert source.seek_time_us == 60000
