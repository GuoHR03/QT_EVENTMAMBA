from dataclasses import dataclass

from backend.event_pipeline import EventPipeline


@dataclass
class CameraRunContext:
    fps: int
    fps_getter: object
    nn_interval_us: int
    replay_factor: float
    replay_factor_getter: object
    is_running: object
    roi_getter: object
    nn_queue: object
    noise_filter: object
    analysis_enabled: object
    roi_snapshot_getter: object = None
    inference_publisher: object = None
    inference_generation_is_current: object = None
    progress_callback: object = None


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
