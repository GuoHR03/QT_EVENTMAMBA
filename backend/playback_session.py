from threading import Event, Lock

from backend.camera_source_runner import close_camera_source, run_camera_source


class PlaybackSession:
    """Own one source playback lifecycle independently of Qt."""

    def __init__(self, source, context, inference_worker=None):
        self.source = source
        self.context = context
        self.inference_worker = inference_worker
        self._running = Event()
        self._running.set()
        self._stop_requested = Event()
        self._worker_started = False
        self._worker_stop_requested = False
        self._worker_stop_lock = Lock()
        self.context.is_running = self.is_running

    def is_running(self):
        return self._running.is_set()

    def run(self):
        completed_naturally = False
        try:
            if not self.is_running():
                return

            metadata = self.source.metadata()
            self.context.noise_filter.initialize(metadata.width, metadata.height)
            self._start_worker()
            if self.is_running():
                run_camera_source(self.source, self.context)
                completed_naturally = not self._stop_requested.is_set()
        finally:
            self._running.clear()
            self._request_worker_stop(
                discard_pending=self._stop_requested.is_set() or not completed_naturally,
            )
            self._join_worker()
            close_camera_source(self.source)

    def stop(self):
        if self._stop_requested.is_set():
            return
        self._stop_requested.set()
        self._running.clear()
        request_stop = getattr(self.source, "request_stop", None)
        if callable(request_stop):
            request_stop()
        self._request_worker_stop(discard_pending=True)

    def _start_worker(self):
        if self.inference_worker is None or self._worker_stop_requested:
            return
        self.inference_worker.start()
        self._worker_started = True

    def _request_worker_stop(self, discard_pending):
        if self.inference_worker is None:
            return
        with self._worker_stop_lock:
            if self._worker_stop_requested:
                return
            self._worker_stop_requested = True
        stop = getattr(self.inference_worker, "stop", None)
        if callable(stop):
            stop(discard_pending=discard_pending)

    def _join_worker(self):
        if not self._worker_started or self.inference_worker is None:
            return
        wait = getattr(self.inference_worker, "wait", None)
        if callable(wait):
            wait()
