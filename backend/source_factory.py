import logging

from backend.event_frame_renderer import EventFrameRenderer
from backend.event_source import Aedat4Source, H5Source, MetavisionSource, SourceMetadata
from backend.h5_source import h5_event_dtype_names, h5_event_time_range, h5_resolution, open_h5_events
from backend.metavision_source import apply_hardware_roi, create_metavision_iterator
from backend.raw_metadata import raw_duration_from_sidecar
from backend.renderer_factory import create_metavision_renderer
from backend.replay_clock import frame_interval_us
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH
from backend.source_metadata import (
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    aedat4_resolution,
    aedat4_time_range,
    classify_input_source,
)


LOGGER = logging.getLogger(__name__)


def create_event_source(
    input_path,
    fps,
    palette_type,
    frame_callback,
    replay_factor=1.0,
    hardware_roi=None,
    status_callback=None,
    replay_factor_getter=None,
    seek_fraction=0.0,
    duration_hint_us=0,
):
    source_type = classify_input_source(input_path)
    if source_type == SOURCE_AEDAT4:
        return create_aedat4_source(
            input_path,
            palette_type,
            fps=fps,
            frame_callback=frame_callback,
            seek_fraction=seek_fraction,
        )
    if source_type == SOURCE_H5:
        return create_h5_source(
            input_path,
            fps,
            palette_type,
            frame_callback,
            seek_fraction=seek_fraction,
        )
    return create_metavision_source(
        input_path=input_path,
        delta_t_us=frame_interval_us(fps),
        replay_factor=replay_factor,
        fps=fps,
        palette_type=palette_type,
        frame_callback=frame_callback,
        hardware_roi=hardware_roi,
        status_callback=status_callback,
        replay_factor_getter=replay_factor_getter,
        seek_fraction=seek_fraction,
        duration_hint_us=duration_hint_us,
    )


def create_aedat4_source(input_path, palette_type, fps=None, frame_callback=None, seek_fraction=0.0):
    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(input_path)
    width, height = aedat4_resolution(reader)
    start_time_us, end_time_us = aedat4_time_range(reader)
    renderer = EventFrameRenderer(width, height, palette_type, frame_callback)
    metadata = SourceMetadata(
        source_type=SOURCE_AEDAT4,
        width=width,
        height=height,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        seekable=end_time_us > start_time_us,
    )
    source = Aedat4Source(reader=reader, renderer=renderer, metadata=metadata)
    source.seek(seek_fraction)
    return source


def create_h5_source(input_path, fps, palette_type, frame_callback, seek_fraction=0.0):
    import h5py

    h5_file = h5py.File(input_path, "r")
    events_dataset = open_h5_events(h5_file)
    dtype_names = h5_event_dtype_names(events_dataset)
    width, height = h5_resolution(h5_file, events_dataset)
    start_time_us, end_time_us = h5_event_time_range(events_dataset, dtype_names)
    renderer = create_metavision_renderer(width, height, fps, palette_type, frame_callback)
    metadata = SourceMetadata(
        source_type=SOURCE_H5,
        width=width,
        height=height,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        seekable=end_time_us > start_time_us,
    )
    source = H5Source(
        source_file=h5_file,
        events_dataset=events_dataset,
        dtype_names=dtype_names,
        renderer=renderer,
        metadata=metadata,
    )
    source.seek(seek_fraction)
    return source


def create_metavision_source(
    input_path,
    delta_t_us,
    replay_factor,
    fps,
    palette_type,
    frame_callback,
    hardware_roi=None,
    status_callback=None,
    replay_factor_getter=None,
    seek_fraction=0.0,
    duration_hint_us=0,
):
    device = None
    start_time_us = 0
    end_time_us = 0
    seek_time_us = 0
    if input_path:
        end_time_us = raw_duration_from_sidecar(input_path) or int(duration_hint_us or 0)
        seek_metadata = SourceMetadata(
            source_type=SOURCE_METAVISION,
            width=DEFAULT_SENSOR_WIDTH,
            height=DEFAULT_SENSOR_HEIGHT,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            seekable=end_time_us > start_time_us,
        )
        seek_time_us = seek_metadata.timestamp_at_fraction(seek_fraction, alignment_us=delta_t_us)
        iterator = create_metavision_iterator(
            input_path,
            device,
            delta_t_us,
            replay_factor,
            replay_factor_getter,
            start_ts=seek_time_us,
        )
    else:
        from metavision_core.event_io.raw_reader import initiate_device

        try:
            device = initiate_device("")
        except Exception as exc:
            LOGGER.exception("Failed to connect camera: %s", exc)
            return None

        if device is None:
            LOGGER.warning("No camera connection and no input file")
            return None

        try:
            apply_hardware_roi(device, hardware_roi, status_callback)
        except Exception as exc:
            LOGGER.exception("Failed to apply hardware ROI: %s", exc)
            if status_callback is not None:
                status_callback(f"[ROI] Failed to apply hardware ROI: {exc}")

        iterator = create_metavision_iterator(
            "", device, delta_t_us, replay_factor, replay_factor_getter
        )

    height, width = iterator.get_size()
    renderer = create_metavision_renderer(width, height, fps, palette_type, frame_callback)
    metadata = SourceMetadata(
        source_type=SOURCE_METAVISION,
        width=width,
        height=height,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        seekable=bool(input_path) and end_time_us > start_time_us,
    )
    source = MetavisionSource(
        device=device,
        iterator=iterator,
        renderer=renderer,
        metadata=metadata,
        seek_alignment_us=delta_t_us,
    )
    source.seek_time_us = seek_time_us
    return source
