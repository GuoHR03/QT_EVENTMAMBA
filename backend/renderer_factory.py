import logging
from threading import Lock

from backend.frame_renderer import FrameRenderer
from backend.palettes import metavision_palette
from backend.replay_clock import frame_interval_us, normalize_fps


LOGGER = logging.getLogger(__name__)


def create_metavision_renderer(width, height, fps, palette_type, frame_callback):
    return MetavisionFrameRenderer(width, height, fps, palette_type, frame_callback)


class MetavisionFrameRenderer(FrameRenderer):
    def __init__(self, width, height, fps, palette_type, frame_callback, generator_factory=None):
        self.width = int(width)
        self.height = int(height)
        self.fps = normalize_fps(fps)
        self.palette_type = palette_type
        self.frame_callback = frame_callback
        self._lock = Lock()
        self._closed = False
        self._generator_factory = generator_factory or _create_periodic_frame_generator
        self._generator = self._build_generator()

    def _build_generator(self):
        return self._generator_factory(
            self.width,
            self.height,
            self.fps,
            self.palette_type,
            self.frame_callback,
        )

    def process_events(self, events):
        with self._lock:
            generator = self._generator
        if generator is not None:
            generator.process_events(events)

    def set_display_settings(self, palette_type=None, fps=None):
        next_palette = palette_type or self.palette_type
        next_fps = normalize_fps(fps if fps is not None else self.fps)
        with self._lock:
            if self._closed:
                return False
            if next_palette == self.palette_type and next_fps == self.fps:
                return False
            self.palette_type = next_palette
            self.fps = next_fps
            self._generator = self._build_generator()
        return True

    def reset(self):
        with self._lock:
            if self._closed:
                return False
            self._generator = self._build_generator()
        return True

    def close(self):
        with self._lock:
            self._closed = True
            self._generator = None
            self.frame_callback = None


DynamicMetavisionFrameGenerator = MetavisionFrameRenderer
create_metavision_frame_generator = create_metavision_renderer


def _create_periodic_frame_generator(width, height, fps, palette_type, frame_callback):
    from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm

    palette = metavision_palette(ColorPalette, palette_type)
    display_fps = normalize_fps(fps)
    accumulation_time_us = frame_interval_us(display_fps)
    frame_generator = _instantiate_periodic_frame_generator(
        PeriodicFrameGenerationAlgorithm,
        width,
        height,
        display_fps,
        accumulation_time_us,
        palette,
    )
    if frame_callback is not None:
        frame_generator.set_output_callback(frame_callback)
    return frame_generator


def _instantiate_periodic_frame_generator(
    frame_generator_cls,
    width,
    height,
    fps,
    accumulation_time_us,
    palette,
):
    kwargs = {
        "sensor_width": width,
        "sensor_height": height,
        "accumulation_time_us": accumulation_time_us,
        "fps": fps,
        "palette": palette,
    }
    try:
        return frame_generator_cls(**kwargs)
    except TypeError:
        kwargs.pop("accumulation_time_us")
        frame_generator = frame_generator_cls(**kwargs)
        _set_accumulation_time_if_supported(frame_generator, accumulation_time_us)
        return frame_generator


def _set_accumulation_time_if_supported(frame_generator, accumulation_time_us):
    for method_name in ("set_accumulation_time_us", "set_accumulation_time"):
        method = getattr(frame_generator, method_name, None)
        if method is None:
            continue
        method(accumulation_time_us)
        return True

    LOGGER.warning(
        "PeriodicFrameGenerationAlgorithm does not expose accumulation time; "
        "using SDK default instead of %sus",
        accumulation_time_us,
    )
    return False
