from dataclasses import dataclass, field

from backend.playback_config import PlaybackConfig


@dataclass
class AppSettings:
    prediction_mode: str = "center"
    playback_config: PlaybackConfig = field(default_factory=PlaybackConfig)

    @property
    def roi(self):
        return self.playback_config.roi

    @property
    def noise_filter_type(self):
        return self.playback_config.noise_filter_type

    @property
    def noise_filter_threshold_us(self):
        return self.playback_config.noise_filter_threshold_us

    @property
    def palette(self):
        return self.playback_config.palette

    @property
    def fps(self):
        return self.playback_config.fps

    @property
    def replay_factor(self):
        return self.playback_config.replay_factor

    def update_capture(self, palette, fps, replay_factor=1.0):
        self.playback_config = self.playback_config.with_updates(
            palette=palette,
            fps=fps,
            replay_factor=replay_factor,
        )

    def update_prediction(self, mode):
        self.prediction_mode = mode

    def update_noise_filter(self, filter_type, threshold_us):
        self.playback_config = self.playback_config.with_updates(
            noise_filter_type=filter_type,
            noise_filter_threshold_us=threshold_us,
        )

    def update_roi(self, roi):
        self.playback_config = self.playback_config.with_updates(roi=roi)
