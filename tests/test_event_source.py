from types import SimpleNamespace

from backend.event_source import Aedat4Source, H5Source, MetavisionSource, SourceMetadata


class ClosableFile:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class StoppableResource:
    def __init__(self):
        self.stop_count = 0

    def request_stop(self):
        self.stop_count += 1


class ClosableRenderer:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_source_metadata_reports_duration_and_aligned_seek_timestamp():
    metadata = SourceMetadata(
        source_type="metavision",
        width=640,
        height=480,
        start_time_us=100,
        end_time_us=100100,
        seekable=True,
    )

    assert metadata.duration_us == 100000
    assert metadata.timestamp_at_fraction(0.5, alignment_us=20000) == 40100


def test_event_source_exposes_common_metadata_and_seek_interface():
    metadata = SourceMetadata("aedat4", 346, 260, 1000, 101000, True)
    source = Aedat4Source(reader=object(), renderer=object(), metadata=metadata)

    assert source.metadata() is metadata
    assert source.source_type == "aedat4"
    assert source.width == 346
    assert source.height == 260
    assert source.seek(0.25) == 26000


def test_h5_source_close_is_idempotent():
    source_file = ClosableFile()
    renderer = ClosableRenderer()
    source = H5Source(
        source_file=source_file,
        events_dataset=object(),
        dtype_names=("x", "y", "p", "t"),
        renderer=renderer,
        metadata=SourceMetadata("h5", 640, 480),
    )

    source.close()
    source.close()

    assert source_file.close_count == 1
    assert renderer.close_count == 1


def test_aedat4_source_delegates_run_with_common_context(monkeypatch):
    calls = {}
    source = Aedat4Source(
        reader=object(),
        renderer=object(),
        metadata=SourceMetadata("aedat4", 640, 480, 0, 100000, True),
    )
    source.seek(0.5)
    context = _context()
    pipeline = object()

    monkeypatch.setattr(
        "backend.event_source.run_aedat4_replay_loop",
        lambda **kwargs: calls.update(kwargs),
    )

    source.run(context, pipeline)

    assert calls["reader"] is source.reader
    assert calls["event_pipeline"] is pipeline
    assert calls["start_time_us"] == 50000


def test_h5_source_delegates_run_with_common_context(monkeypatch):
    calls = {}
    source = H5Source(
        source_file=ClosableFile(),
        events_dataset=object(),
        dtype_names=("x", "y", "p", "t"),
        renderer=object(),
        metadata=SourceMetadata("h5", 640, 480, 1000, 101000, True),
    )
    source.seek(0.25)
    context = _context()
    pipeline = SimpleNamespace(process_events=lambda events: events)

    monkeypatch.setattr(
        "backend.event_source.run_h5_replay_loop",
        lambda **kwargs: calls.update(kwargs),
    )

    source.run(context, pipeline)

    assert calls["events_dataset"] is source.events_dataset
    assert calls["dtype_names"] == source.dtype_names
    assert calls["start_time_us"] == 26000
    assert calls["handle_frame_events"] is pipeline.process_events


def test_metavision_source_delegates_run_with_common_context(monkeypatch):
    calls = {}
    source = MetavisionSource(
        device=None,
        iterator=object(),
        renderer=object(),
        metadata=SourceMetadata("metavision", 640, 480),
    )
    context = _context()
    pipeline = object()

    monkeypatch.setattr(
        "backend.event_source.run_metavision_event_loop",
        lambda **kwargs: calls.update(kwargs),
    )

    source.run(context, pipeline)

    assert calls["iterator"] is source.iterator
    assert calls["event_pipeline"] is pipeline


def test_metavision_source_uses_delta_alignment_for_seek():
    source = MetavisionSource(
        device=None,
        iterator=object(),
        renderer=object(),
        metadata=SourceMetadata("metavision", 640, 480, 0, 123456, True),
        seek_alignment_us=20000,
    )

    assert source.seek(0.5) == 60000


def test_source_request_stop_delegates_to_underlying_resource():
    reader = StoppableResource()
    iterator = StoppableResource()
    metadata = SourceMetadata("aedat4", 640, 480)
    aedat4_source = Aedat4Source(reader, object(), metadata)
    metavision_source = MetavisionSource(
        device=None,
        iterator=iterator,
        renderer=object(),
        metadata=SourceMetadata("metavision", 640, 480),
    )

    assert aedat4_source.request_stop()
    assert metavision_source.request_stop()
    assert reader.stop_count == 1
    assert iterator.stop_count == 1


def _context():
    return SimpleNamespace(
        fps=30,
        fps_getter=lambda: 30,
        replay_factor=1.0,
        replay_factor_getter=lambda: 1.0,
        is_running=lambda: True,
        progress_callback=None,
    )
