from dataclasses import dataclass
import time

from backend.aedat4_source import run_aedat4_replay_loop
from backend.camera_source_factory import SOURCE_AEDAT4, SOURCE_H5, SOURCE_METAVISION
from backend.frame_pipeline import H5FrameProcessor
from backend.h5_replay import run_h5_replay_loop
from backend.metavision_source import run_metavision_event_loop


@dataclass
class CameraRunContext:
    fps: int
    fps_getter: object
    nn_interval_us: int
    replay_factor: float
    replay_factor_getter: object
    is_running: object
    roi_getter: object
    image_callback: object
    nn_queue: object
    noise_filter: object
    target_queue: object
    analysis_enabled: object
    progress_callback: object = None


def run_camera_source(source_type, source, context):
    if source is None:
        return

    if source_type == SOURCE_AEDAT4:
        _run_aedat4_source(source, context)
    elif source_type == SOURCE_H5:
        _run_h5_source(source, context)
    elif source_type == SOURCE_METAVISION:
        _run_metavision_source(source, context)
    else:
        raise ValueError(f"Unsupported camera source type: {source_type}")


def _run_aedat4_source(source, context):
    run_aedat4_replay_loop(
        reader=source.reader,
        frame_generator=source.frame_generator,
        fps=context.fps,
        nn_interval_us=context.nn_interval_us,
        is_running=context.is_running,
        roi_getter=context.roi_getter,
        nn_queue=context.nn_queue,
        noise_filter=context.noise_filter,
        replay_factor=context.replay_factor,
        replay_factor_getter=context.replay_factor_getter,
        fps_getter=context.fps_getter,
        start_time_us=getattr(source, "seek_time_us", 0),
        progress_callback=context.progress_callback,
    )


def _run_h5_source(source, context):
    processor = H5FrameProcessor(
        width=source.width,
        height=source.height,
        roi_getter=context.roi_getter,
        noise_filter=context.noise_filter,
        frame_generator=source.frame_generator,
        target_queue=context.target_queue,
        analysis_enabled=context.analysis_enabled,
    )
    try:
        run_h5_replay_loop(
            events_dataset=source.events_dataset,
            dtype_names=source.dtype_names,
            fps=context.fps,
            is_running=context.is_running,
            handle_frame_events=processor.handle_frame_events,
            now=time.perf_counter,
            sleep=time.sleep,
            replay_factor=context.replay_factor,
            replay_factor_getter=context.replay_factor_getter,
            fps_getter=context.fps_getter,
            start_time_us=getattr(source, "seek_time_us", 0),
            progress_callback=context.progress_callback,
        )
    finally:
        close_camera_source(source)


def _run_metavision_source(source, context):
    run_metavision_event_loop(
        iterator=source.iterator,
        is_running=context.is_running,
        roi_getter=context.roi_getter,
        noise_filter=context.noise_filter,
        frame_generator=source.frame_generator,
        nn_queue=context.nn_queue,
        progress_callback=context.progress_callback,
    )


def close_camera_source(source):
    source_file = getattr(source, "file", None)
    if source_file is not None:
        source_file.close()
