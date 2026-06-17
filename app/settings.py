from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AppSettings:
    prediction_mode: str = "center"
    roi: Optional[Tuple[int, int, int, int]] = None
    noise_filter_type: str = "none"
    noise_filter_threshold_us: int = 10000
    palette: str = "Dark"
    fps: float = 30.0

    def update_capture(self, palette, fps):
        self.palette = palette
        self.fps = float(fps)

    def update_prediction(self, mode):
        self.prediction_mode = mode

    def update_noise_filter(self, filter_type, threshold_us):
        self.noise_filter_type = filter_type or "none"
        self.noise_filter_threshold_us = int(threshold_us)

    def update_roi(self, roi):
        self.roi = tuple(roi) if roi is not None else None
