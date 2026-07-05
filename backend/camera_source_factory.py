import logging
from dataclasses import dataclass
from threading import Lock

from backend.event_frame_renderer import EventFrameRenderer
from backend.metavision_source import apply_hardware_roi, create_metavision_iterator
from backend.h5_source import h5_event_dtype_names, h5_resolution, open_h5_events
from backend.palettes import metavision_palette
from backend.replay_clock import normalize_fps
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH

LOGGER = logging.getLogger(__name__)

SOURCE_AEDAT4 = "aedat4"
SOURCE_H5 = "h5"
SOURCE_METAVISION = "metavision"


@dataclass
class Aedat4Source:
    reader: object
    frame_generator: object
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


def create_aedat4_source(input_path, palette_type, fps=None, frame_callback=None):
    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(input_path)
    width, height = _aedat4_resolution(reader)
    frame_generator = EventFrameRenderer(width, height, palette_type, frame_callback)

    return Aedat4Source(
        reader=reader,
        frame_generator=frame_generator,
        width=width,
        height=height,
    )


def create_h5_source(input_path, fps, palette_type, frame_callback):
    import h5py

    h5_file = h5py.File(input_path, "r")
    events_dataset = open_h5_events(h5_file)
    width, height = h5_resolution(h5_file, events_dataset)
    frame_generator = create_metavision_frame_generator(width, height, fps, palette_type, frame_callback)
    return H5Source(
        file=h5_file,
        events_dataset=events_dataset,
        dtype_names=h5_event_dtype_names(events_dataset),
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
    replay_factor_getter=None,
):
    device = None
    if input_path:
        iterator = create_metavision_iterator(input_path, device, delta_t_us, replay_factor, replay_factor_getter)
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

        iterator = create_metavision_iterator("", device, delta_t_us, replay_factor, replay_factor_getter)

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
    return DynamicMetavisionFrameGenerator(width, height, fps, palette_type, frame_callback)


class DynamicMetavisionFrameGenerator:
    def __init__(self, width, height, fps, palette_type, frame_callback, generator_factory=None):
        self.width = int(width)
        self.height = int(height)
        self.fps = normalize_fps(fps)
        self.palette_type = palette_type
        self.frame_callback = frame_callback
        self._lock = Lock()
        self._generator_factory = generator_factory or _create_periodic_frame_generator
        self._generator = self._generator_factory(
            self.width,
            self.height,
            self.fps,
            self.palette_type,
            self.frame_callback,
        )

    def process_events(self, events):
        generator = self._generator
        generator.process_events(events)

    def set_display_settings(self, palette_type=None, fps=None):
        next_palette = palette_type or self.palette_type
        next_fps = normalize_fps(fps if fps is not None else self.fps)
        with self._lock:
            if next_palette == self.palette_type and next_fps == self.fps:
                return False
            self.palette_type = next_palette
            self.fps = next_fps
            self._generator = self._generator_factory(
                self.width,
                self.height,
                self.fps,
                self.palette_type,
                self.frame_callback,
            )
        return True


def _create_periodic_frame_generator(width, height, fps, palette_type, frame_callback):
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    palette = metavision_palette(ColorPalette, palette_type)
    frame_generator = PeriodicFrameGenerationAlgorithm(
        sensor_width=width,
        sensor_height=height,
        fps=fps if fps > 0 else 30,
        palette=palette,
    )
    if frame_callback is not None:
        frame_generator.set_output_callback(frame_callback)
    return frame_generator


def _aedat4_resolution(reader):
    try:
        resolution = reader.getEventResolution()
        if isinstance(resolution, tuple):
            return int(resolution[0]), int(resolution[1])
        return int(resolution.width), int(resolution.height)
    except Exception:
        return DEFAULT_SENSOR_WIDTH, DEFAULT_SENSOR_HEIGHT
