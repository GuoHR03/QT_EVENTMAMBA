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
    replay_factor: float = 1.0

    def update_capture(self, palette, fps, replay_factor=1.0):
        self.palette = palette
        self.fps = float(fps)
        self.replay_factor = max(float(replay_factor or 1.0), 0.001)

    def update_prediction(self, mode):
        self.prediction_mode = mode

    def update_noise_filter(self, filter_type, threshold_us):
        self.noise_filter_type = filter_type or "none"
        self.noise_filter_threshold_us = int(threshold_us)

    def update_roi(self, roi):
        self.roi = tuple(roi) if roi is not None else None
