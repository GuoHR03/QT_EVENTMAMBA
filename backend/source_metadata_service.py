import threading

from backend.raw_metadata import compute_raw_duration, raw_duration_from_sidecar


class SourceMetadataService:
    """Cache source durations and resolve missing RAW duration in the background."""

    def __init__(
        self,
        sidecar_reader=None,
        duration_scanner=None,
        worker_factory=None,
    ):
        self._sidecar_reader = sidecar_reader or raw_duration_from_sidecar
        self._duration_scanner = duration_scanner or compute_raw_duration
        self._worker_factory = worker_factory or threading.Thread
        self._duration_cache = {}
        self._duration_scans = set()
        self._lock = threading.Lock()

    def duration_hint(self, input_path):
        input_path = str(input_path or "")
        if not input_path:
            return 0

        with self._lock:
            cached_duration = self._duration_cache.get(input_path, 0)
        if cached_duration > 0:
            return cached_duration
        if not _is_raw_file(input_path):
            return 0

        duration_us = max(0, int(self._sidecar_reader(input_path) or 0))
        if duration_us > 0:
            self.record_duration(input_path, duration_us)
        return duration_us

    def record_duration(self, input_path, duration_us):
        input_path = str(input_path or "")
        duration_us = max(0, int(duration_us or 0))
        if not input_path or duration_us <= 0:
            return False
        with self._lock:
            self._duration_cache[input_path] = duration_us
        return True

    def ensure_duration_scan(self, input_path, callback=None):
        input_path = str(input_path or "")
        if not _is_raw_file(input_path) or self.duration_hint(input_path) > 0:
            return False

        with self._lock:
            if input_path in self._duration_scans:
                return False
            self._duration_scans.add(input_path)

        worker = self._worker_factory(
            target=self._scan_duration,
            args=(input_path, callback),
            name="RawDurationScanner",
            daemon=True,
        )
        worker.start()
        return True

    def _scan_duration(self, input_path, callback):
        try:
            duration_us = max(0, int(self._duration_scanner(input_path) or 0))
            if duration_us > 0:
                self.record_duration(input_path, duration_us)
        finally:
            with self._lock:
                self._duration_scans.discard(input_path)

        if duration_us > 0 and callback is not None:
            callback(input_path, duration_us)


def _is_raw_file(input_path):
    return bool(input_path) and str(input_path).lower().endswith(".raw")
