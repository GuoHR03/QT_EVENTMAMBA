from dataclasses import dataclass
import time

from backend.aedat4_source import run_aedat4_replay_loop
from backend.h5_replay import run_h5_replay_loop
from backend.metavision_source import run_metavision_event_loop


@dataclass(frozen=True)
class SourceMetadata:
    source_type: str
    width: int
    height: int
    start_time_us: int = 0
    end_time_us: int = 0
    seekable: bool = False

    def __post_init__(self):
        object.__setattr__(self, "width", max(1, int(self.width)))
        object.__setattr__(self, "height", max(1, int(self.height)))
        object.__setattr__(self, "start_time_us", max(0, int(self.start_time_us or 0)))
        object.__setattr__(self, "end_time_us", max(0, int(self.end_time_us or 0)))

    @property
    def duration_us(self):
        return max(0, self.end_time_us - self.start_time_us)

    def timestamp_at_fraction(self, fraction, alignment_us=1):
        if not self.seekable or self.duration_us <= 0:
            return self.start_time_us

        fraction = _clamp_fraction(fraction)
        offset_us = int(self.duration_us * fraction)
        alignment_us = max(1, int(alignment_us or 1))
        offset_us = (offset_us // alignment_us) * alignment_us
        return self.start_time_us + offset_us


class EventSource:
    """Common interface for opened event inputs."""

    def __init__(self, metadata, renderer, device=None, seek_alignment_us=1):
        self._metadata = metadata
        self.renderer = renderer
        self.device = device
        self.seek_alignment_us = max(1, int(seek_alignment_us or 1))
        self.seek_time_us = metadata.start_time_us
        self._closed = False

    @property
    def source_type(self):
        return self._metadata.source_type

    @property
    def width(self):
        return self._metadata.width

    @property
    def height(self):
        return self._metadata.height

    @property
    def start_time_us(self):
        return self._metadata.start_time_us

    @property
    def end_time_us(self):
        return self._metadata.end_time_us

    def metadata(self):
        return self._metadata

    def seek(self, fraction):
        self.seek_time_us = self._metadata.timestamp_at_fraction(
            fraction,
            alignment_us=self.seek_alignment_us,
        )
        return self.seek_time_us

    def run(self, context, event_pipeline):
        raise NotImplementedError

    def request_stop(self):
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self.renderer, "close", None)
        if callable(close):
            close()


class Aedat4Source(EventSource):
    def __init__(self, reader, renderer, metadata):
        super().__init__(metadata, renderer)
        self.reader = reader

    def run(self, context, event_pipeline):
        run_aedat4_replay_loop(
            reader=self.reader,
            event_pipeline=event_pipeline,
            fps=context.fps,
            is_running=context.is_running,
            replay_factor=context.replay_factor,
            replay_factor_getter=context.replay_factor_getter,
            fps_getter=context.fps_getter,
            start_time_us=self.seek_time_us,
            progress_callback=context.progress_callback,
        )

    def request_stop(self):
        return _request_resource_stop(self.reader)


class H5Source(EventSource):
    def __init__(self, source_file, events_dataset, dtype_names, renderer, metadata):
        super().__init__(metadata, renderer)
        self.file = source_file
        self.events_dataset = events_dataset
        self.dtype_names = dtype_names

    def run(self, context, event_pipeline):
        run_h5_replay_loop(
            events_dataset=self.events_dataset,
            dtype_names=self.dtype_names,
            fps=context.fps,
            is_running=context.is_running,
            handle_frame_events=event_pipeline.process_events,
            now=time.perf_counter,
            sleep=time.sleep,
            replay_factor=context.replay_factor,
            replay_factor_getter=context.replay_factor_getter,
            fps_getter=context.fps_getter,
            start_time_us=self.seek_time_us,
            progress_callback=context.progress_callback,
        )

    def close(self):
        if self._closed:
            return
        try:
            if self.file is not None:
                self.file.close()
        finally:
            super().close()


class MetavisionSource(EventSource):
    def __init__(self, device, iterator, renderer, metadata, seek_alignment_us=1):
        super().__init__(metadata, renderer, device=device, seek_alignment_us=seek_alignment_us)
        self.iterator = iterator

    def run(self, context, event_pipeline):
        run_metavision_event_loop(
            iterator=self.iterator,
            is_running=context.is_running,
            event_pipeline=event_pipeline,
            progress_callback=context.progress_callback,
        )

    def request_stop(self):
        return _request_resource_stop(self.iterator)


def _request_resource_stop(resource):
    if resource is None:
        return False
    for method_name in ("request_stop", "requestStop", "interrupt"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            method()
            return True
        except Exception:
            return False
    return False


def _clamp_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, fraction))
