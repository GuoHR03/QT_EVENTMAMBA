import logging
from dataclasses import dataclass

from backend.metavision_source import apply_hardware_roi, create_metavision_iterator
from backend.palettes import apply_aedat4_palette, metavision_palette

LOGGER = logging.getLogger(__name__)

SOURCE_AEDAT4 = "aedat4"
SOURCE_H5 = "h5"
SOURCE_METAVISION = "metavision"


@dataclass
class Aedat4Source:
    reader: object
    visualizer: object
    width: int
    height: int
    device: object = None


@dataclass
class H5Source:
    file: object
    events_dataset: object
    dtype_names: tuple
    frame_generator: object
    width: int
    height: int
    device: object = None


@dataclass
class MetavisionSource:
    device: object
    iterator: object
    frame_generator: object
    width: int
    height: int


def classify_input_source(input_path):
    path = (input_path or "").lower()
    if path.endswith(".aedat4"):
        return SOURCE_AEDAT4
    if path.endswith((".h5", ".hdf5")):
        return SOURCE_H5
    return SOURCE_METAVISION


def create_aedat4_source(input_path, palette_type):
    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(input_path)
    width, height = _aedat4_resolution(reader)
    visualizer = dv.visualization.EventVisualizer((width, height))
    apply_aedat4_palette(visualizer, palette_type)
    return Aedat4Source(reader=reader, visualizer=visualizer, width=width, height=height)


def create_h5_source(input_path, fps, palette_type, frame_callback):
    import h5py

    h5_file = h5py.File(input_path, "r")
    events_dataset = h5_file["events"]
    width = int(h5_file.attrs.get("width", 640))
    height = int(h5_file.attrs.get("height", 480))
    frame_generator = create_metavision_frame_generator(width, height, fps, palette_type, frame_callback)
    return H5Source(
        file=h5_file,
        events_dataset=events_dataset,
        dtype_names=events_dataset.dtype.names,
        frame_generator=frame_generator,
        width=width,
        height=height,
    )


def create_metavision_source(
    input_path,
    delta_t_us,
    replay_factor,
    fps,
    palette_type,
    frame_callback,
    hardware_roi=None,
    status_callback=None,
):
    device = None
    if input_path:
        iterator = create_metavision_iterator(input_path, device, delta_t_us, replay_factor)
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

        iterator = create_metavision_iterator("", device, delta_t_us, replay_factor)

    height, width = iterator.get_size()
    frame_generator = create_metavision_frame_generator(width, height, fps, palette_type, frame_callback)
    return MetavisionSource(
        device=device,
        iterator=iterator,
        frame_generator=frame_generator,
        width=width,
        height=height,
    )


def create_metavision_frame_generator(width, height, fps, palette_type, frame_callback):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    palette = metavision_palette(ColorPalette, palette_type)
    frame_generator = PeriodicFrameGenerationAlgorithm(
        sensor_width=width,
        sensor_height=height,
        fps=fps if fps > 0 else 30,
        palette=palette,
    )
    frame_generator.set_output_callback(frame_callback)
    return frame_generator


def _aedat4_resolution(reader):
    try:
        resolution = reader.getEventResolution()
        return int(resolution.width), int(resolution.height)
    except Exception:
        return 640, 480
