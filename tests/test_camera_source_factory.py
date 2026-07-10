from backend.camera_source_factory import (
    Aedat4Source,
    DynamicMetavisionFrameGenerator,
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    classify_input_source,
    create_metavision_source,
)


def test_classify_input_source_detects_aedat4():
    assert classify_input_source("sample.AEDAT4") == SOURCE_AEDAT4


def test_classify_input_source_detects_h5_variants():
    assert classify_input_source("sample.h5") == SOURCE_H5
    assert classify_input_source("sample.HDF5") == SOURCE_H5


def test_classify_input_source_defaults_to_metavision():
    assert classify_input_source("") == SOURCE_METAVISION
    assert classify_input_source("recording.raw") == SOURCE_METAVISION


def test_aedat4_source_records_frame_generator():
    source = Aedat4Source(
        reader=object(),
        frame_generator=object(),
        width=640,
        height=480,
    )

    assert source.frame_generator is not None


def test_dynamic_metavision_frame_generator_rebuilds_renderer_on_display_update():
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

    frame_generator = DynamicMetavisionFrameGenerator(
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

    assert changed is True
    assert len(created) == 2
    assert created[0].events == ["before"]
    assert created[1].palette_type == "Light"
    assert created[1].fps == 60
    assert created[1].events == ["after"]


def test_metavision_source_uses_duration_hint_for_raw_seek(monkeypatch):
    calls = {}

    class FakeIterator:
        def get_size(self):
            return 480, 640

    def fake_iterator(input_path, device, delta_t_us, replay_factor, replay_factor_getter=None, start_ts=0):
        calls["input_path"] = input_path
        calls["start_ts"] = start_ts
        return FakeIterator()

    monkeypatch.setattr("backend.camera_source_factory.create_metavision_iterator", fake_iterator)
    monkeypatch.setattr("backend.camera_source_factory.create_metavision_frame_generator", lambda *args: object())

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
