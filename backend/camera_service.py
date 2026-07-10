import threading

from backend.Camera import CameraThread
from backend.raw_metadata import compute_raw_duration, raw_duration_from_sidecar
from backend.settings import DEFAULT_FPS, DEFAULT_NOISE_FILTER_THRESHOLD_US


class CameraService:
    def __init__(self, frame_queue, image_signal, status_signal, finished_signal, progress_signal):
        self.frame_queue = frame_queue
        self.image_signal = image_signal
        self.status_signal = status_signal
        self.finished_signal = finished_signal
        self.progress_signal = progress_signal
        self.thread = None
        self.file_path = None
        self.last_palette = "Dark"
        self.last_fps = DEFAULT_FPS
        self.last_replay_factor = 1.0
        self.last_seek_fraction = 0.0
        self._duration_cache = {}
        self._duration_scans = set()
        self._duration_lock = threading.Lock()
        self._last_progress_current_us = 0

    def is_running(self):
        return self.thread is not None and self.thread.isRunning()

    def is_recording(self):
        return self.is_running() and self.thread.is_recording

    def set_input_file(self, file_path):
        self.file_path = file_path
        self.last_seek_fraction = 0.0

    def start(
        self,
        palette,
        fps,
        roi=None,
        replay_factor=1.0,
        noise_filter_type="none",
        noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US,
        seek_fraction=0.0,
        report_noise_filter_status=True,
    ):
        self.last_palette = palette
        self.last_fps = fps
        self.last_replay_factor = replay_factor
        self.last_seek_fraction = _clamp_fraction(seek_fraction)
        if self.thread is not None:
            self.stop(emit_finished=False)

        kwargs = {
            "palette_type": palette,
            "fps": fps,
            "replay_factor": replay_factor,
            "target_queue": self.frame_queue,
            "noise_filter_type": noise_filter_type,
            "noise_filter_threshold_us": noise_filter_threshold_us,
            "seek_fraction": self.last_seek_fraction,
            "duration_hint_us": self._duration_hint_for_current_file(),
            "report_noise_filter_status": report_noise_filter_status,
        }
        if self.file_path:
            kwargs["file_path"] = self.file_path
        if roi:
            kwargs["roi"] = roi

        self.thread = CameraThread(**kwargs)
        self.thread.image_signal.connect(self.image_signal.emit)
        self.thread.status_signal.connect(self.status_signal.emit)
        self.thread.finished_signal.connect(self.finished_signal.emit)
        self.thread.progress_signal.connect(self._forward_progress)
        self.thread.start()
        self._ensure_raw_duration_scan()

    def restart(
        self,
        palette=None,
        fps=None,
        roi=None,
        replay_factor=None,
        noise_filter_type="none",
        noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US,
        seek_fraction=None,
        report_noise_filter_status=True,
    ):
        palette = palette if palette is not None else self.last_palette
        fps = fps if fps is not None else self.last_fps
        replay_factor = replay_factor if replay_factor is not None else self.last_replay_factor
        seek_fraction = self.last_seek_fraction if seek_fraction is None else seek_fraction
        self.stop(emit_finished=False)
        self.start(
            palette,
            fps,
            roi,
            replay_factor,
            noise_filter_type,
            noise_filter_threshold_us,
            seek_fraction,
            report_noise_filter_status,
        )

    def seek(self, seek_fraction, noise_filter_type="none", noise_filter_threshold_us=DEFAULT_NOISE_FILTER_THRESHOLD_US):
        roi = self.current_roi()
        self.restart(
            self.last_palette,
            self.last_fps,
            roi,
            self.last_replay_factor,
            noise_filter_type,
            noise_filter_threshold_us,
            seek_fraction=seek_fraction,
            report_noise_filter_status=False,
        )

    def set_replay_factor(self, replay_factor):
        self.last_replay_factor = replay_factor
        if self.thread is not None:
            self.thread.set_replay_factor(replay_factor)

    def set_display_settings(self, palette, fps):
        self.last_palette = palette
        self.last_fps = fps
        if self.thread is not None:
            self.thread.set_display_settings(palette, fps)

    def stop(self, emit_finished=True):
        if not self.thread:
            return

        if not emit_finished:
            try:
                self.thread.finished_signal.disconnect(self.finished_signal.emit)
            except (TypeError, RuntimeError):
                pass
        try:
            self.thread.progress_signal.disconnect(self._forward_progress)
        except (TypeError, RuntimeError):
            pass

        self.thread.stop()
        if not self.thread.wait(1000):
            self.thread.terminate()
            self.thread.wait(500)

        self.thread.deleteLater()
        self.thread = None

    def start_recording(self):
        if self.is_running():
            self.thread.start_recording()

    def stop_recording(self):
        if self.is_running():
            self.thread.stop_recording()

    def current_roi(self):
        if not self.thread:
            return None
        return self.thread.requested_roi

    def current_size(self):
        if not self.thread:
            return None
        return self.thread.width, self.thread.height

    def _duration_hint_for_current_file(self):
        if not self.file_path:
            return 0
        cached_duration = self._duration_cache.get(self.file_path, 0)
        if cached_duration > 0:
            return cached_duration
        sidecar_duration = raw_duration_from_sidecar(self.file_path)
        if sidecar_duration > 0:
            self._duration_cache[self.file_path] = sidecar_duration
        return sidecar_duration

    def _forward_progress(self, current_us, total_us):
        current_us = max(0, int(current_us or 0))
        total_us = max(0, int(total_us or 0))
        self._last_progress_current_us = current_us
        if total_us > 0 and self.file_path:
            self._duration_cache[self.file_path] = total_us
        self.progress_signal.emit(current_us, total_us)

    def _ensure_raw_duration_scan(self):
        if not self.file_path or not self.file_path.lower().endswith(".raw"):
            return
        if self._duration_hint_for_current_file() > 0:
            return

        input_path = self.file_path
        with self._duration_lock:
            if input_path in self._duration_scans:
                return
            self._duration_scans.add(input_path)

        worker = threading.Thread(
            target=self._scan_raw_duration,
            args=(input_path,),
            name="RawDurationScanner",
            daemon=True,
        )
        worker.start()

    def _scan_raw_duration(self, input_path):
        duration_us = compute_raw_duration(input_path)
        with self._duration_lock:
            self._duration_scans.discard(input_path)

        if duration_us <= 0:
            return

        self._duration_cache[input_path] = duration_us
        if self.file_path != input_path:
            return

        current_us = min(max(0, self._last_progress_current_us), duration_us)
        self.progress_signal.emit(current_us, duration_us)


def _clamp_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, fraction))
