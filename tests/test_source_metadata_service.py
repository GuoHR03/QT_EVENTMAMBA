from backend.source_metadata_service import SourceMetadataService


class DeferredThread:
    def __init__(self, target, args, name, daemon):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def run(self):
        self.target(*self.args)


def test_metadata_service_uses_and_caches_raw_sidecar_duration():
    calls = []
    service = SourceMetadataService(
        sidecar_reader=lambda path: calls.append(path) or 1234,
        duration_scanner=lambda _path: 9999,
    )

    assert service.duration_hint("events.raw") == 1234
    assert service.duration_hint("events.raw") == 1234
    assert calls == ["events.raw"]
    assert not service.ensure_duration_scan("events.raw")


def test_metadata_service_scans_missing_raw_duration_once_and_notifies():
    workers = []
    callbacks = []

    def worker_factory(**kwargs):
        worker = DeferredThread(**kwargs)
        workers.append(worker)
        return worker

    service = SourceMetadataService(
        sidecar_reader=lambda _path: 0,
        duration_scanner=lambda _path: 5678,
        worker_factory=worker_factory,
    )

    callback = lambda path, duration: callbacks.append((path, duration))
    assert service.ensure_duration_scan("events.raw", callback)
    assert not service.ensure_duration_scan("events.raw", callback)
    assert len(workers) == 1
    assert workers[0].started

    workers[0].run()

    assert callbacks == [("events.raw", 5678)]
    assert service.duration_hint("events.raw") == 5678


def test_metadata_service_does_not_scan_non_raw_files():
    sidecar_calls = []
    service = SourceMetadataService(
        sidecar_reader=lambda path: sidecar_calls.append(path) or 100,
    )

    assert service.duration_hint("events.h5") == 0
    assert not service.ensure_duration_scan("events.aedat4")
    assert sidecar_calls == []


def test_metadata_service_accepts_duration_reported_by_source():
    service = SourceMetadataService(sidecar_reader=lambda _path: 0)

    assert service.record_duration("events.raw", 4321)
    assert service.duration_hint("events.raw") == 4321
    assert not service.record_duration("events.raw", 0)
