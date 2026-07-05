from backend.camera_source_factory import (
    Aedat4Source,
    DynamicMetavisionFrameGenerator,
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    classify_input_source,
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
