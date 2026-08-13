from threading import Lock

import numpy as np

from backend.frame_renderer import FrameRenderer
from backend.palettes import aedat4_rgb_palette


class EventFrameRenderer(FrameRenderer):
    """Render CD events into a BGR image using palette colors."""

    def __init__(self, width, height, palette_type="Dark", frame_callback=None):
        self.width = int(width)
        self.height = int(height)
        self.frame_callback = frame_callback
        self.palette_type = palette_type
        self._lock = Lock()
        self._set_palette_colors(palette_type)
        self._frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._closed = False

    def set_display_settings(self, palette_type=None, fps=None):
        if palette_type is None or palette_type == self.palette_type:
            return False
        with self._lock:
            self.palette_type = palette_type
            self._set_palette_colors(palette_type)
        return True

    def process_events(self, events):
        if events is None or len(events) == 0:
            return

        with self._lock:
            if self._closed or self.frame_callback is None:
                return
            self._frame[:, :] = self.background

            x = np.asarray(events["x"], dtype=np.int64)
            y = np.asarray(events["y"], dtype=np.int64)
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            if not np.any(valid):
                return

            x = x[valid]
            y = y[valid]
            polarity = np.asarray(events["p"])[valid]
            keep = _last_event_per_pixel(x, y, self.width)
            x = x[keep]
            y = y[keep]
            polarity = polarity[keep]

            positive = polarity > 0
            if np.any(~positive):
                self._frame[y[~positive], x[~positive]] = self.negative
            if np.any(positive):
                self._frame[y[positive], x[positive]] = self.positive

            callback = self.frame_callback
            frame = self._frame.copy()
        callback(int(events["t"][-1]), frame)

    def reset(self):
        if self._closed:
            return False
        with self._lock:
            self._frame[:, :] = self.background
        return True

    def close(self):
        with self._lock:
            self._closed = True
            self.frame_callback = None

    def _set_palette_colors(self, palette_type):
        palette = aedat4_rgb_palette(palette_type)
        self.background = _rgb_to_bgr(palette["bg"])
        self.positive = _rgb_to_bgr(palette["pos"])
        self.negative = _rgb_to_bgr(palette["neg"])


def _rgb_to_bgr(color):
    r, g, b = color
    return np.array((b, g, r), dtype=np.uint8)


def _last_event_per_pixel(x, y, width):
    flat = y * width + x
    _, reversed_indices = np.unique(flat[::-1], return_index=True)
    keep = len(flat) - 1 - reversed_indices
    return np.sort(keep)
