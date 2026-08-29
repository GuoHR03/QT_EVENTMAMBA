from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple

from backend.event_pipeline import EventPipeline


Roi = Optional[Tuple[int, int, int, int]]
RoiSnapshot = Tuple[Any, Roi]


class NoiseFilterPort(Protocol):
    def apply(self, events):
        ...


@dataclass
class CameraRunContext:
    """Typed runtime dependencies passed from playback to an event source."""

    fps: float
    fps_getter: Callable[[], float]
    nn_interval_us: int
    replay_factor: float
    replay_factor_getter: Callable[[], float]
    is_running: Callable[[], bool]
    roi_getter: Callable[[], Roi]
    nn_queue: Any
    noise_filter: NoiseFilterPort
    analysis_enabled: Callable[[], bool]
    roi_snapshot_getter: Optional[Callable[[], RoiSnapshot]] = None
    inference_publisher: Optional[Callable[[Any, Any], bool]] = None
    inference_generation_is_current: Optional[Callable[[Any], bool]] = None
    progress_callback: Optional[Callable[[int, int], None]] = None


def run_camera_source(source, context):
    if source is None:
        return

    pipeline = _create_event_pipeline(source, context)
    source.run(context, pipeline)


def _create_event_pipeline(source, context):
    return EventPipeline(
        roi_getter=context.roi_getter,
        noise_filter=context.noise_filter,
        renderer=source.renderer,
        inference_queue=context.nn_queue,
        inference_interval_us=context.nn_interval_us,
        analysis_enabled=context.analysis_enabled,
        roi_snapshot_getter=context.roi_snapshot_getter,
        inference_publisher=context.inference_publisher,
        inference_generation_is_current=context.inference_generation_is_current,
    )


def close_camera_source(source):
    close = getattr(source, "close", None)
    if callable(close):
        close()
        return

    source_file = getattr(source, "file", None)
    if source_file is not None:
        source_file.close()
