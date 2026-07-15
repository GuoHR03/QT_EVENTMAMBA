from dataclasses import dataclass, replace
from threading import Lock

from backend.event_processing import normalize_noise_filter_type
from backend.replay_clock import normalize_fps
from backend.replay_speed import normalize_replay_factor
from backend.settings import (
    DEFAULT_FPS,
    DEFAULT_NN_INTERVAL_MS,
    DEFAULT_NOISE_FILTER_THRESHOLD_US,
    DEFAULT_REPLAY_FACTOR,
)


@dataclass(frozen=True)
class PlaybackConfig:
    palette: str = "Dark"
    fps: float = DEFAULT_FPS
    replay_factor: float = DEFAULT_REPLAY_FACTOR
    roi: tuple = None
    noise_filter_type: str = "none"
    noise_filter_threshold_us: int = DEFAULT_NOISE_FILTER_THRESHOLD_US
    nn_interval_ms: int = DEFAULT_NN_INTERVAL_MS

    def __post_init__(self):
        object.__setattr__(self, "palette", str(self.palette or "Dark"))
        object.__setattr__(self, "fps", normalize_fps(self.fps))
        object.__setattr__(self, "replay_factor", normalize_replay_factor(self.replay_factor))
        object.__setattr__(self, "roi", _normalize_roi_tuple(self.roi))
        object.__setattr__(self, "noise_filter_type", normalize_noise_filter_type(self.noise_filter_type))
        object.__setattr__(
            self,
            "noise_filter_threshold_us",
            max(1, int(self.noise_filter_threshold_us or DEFAULT_NOISE_FILTER_THRESHOLD_US)),
        )
        object.__setattr__(self, "nn_interval_ms", max(1, int(self.nn_interval_ms or DEFAULT_NN_INTERVAL_MS)))

    @property
    def nn_interval_us(self):
        return self.nn_interval_ms * 1000

    def with_updates(self, **changes):
        return replace(self, **changes)


class PlaybackConfigController:
    def __init__(self, config=None):
        self._lock = Lock()
        self._config = config or PlaybackConfig()

    def get(self):
        with self._lock:
            return self._config

    def set(self, config):
        if not isinstance(config, PlaybackConfig):
            raise TypeError("config must be a PlaybackConfig")
        with self._lock:
            previous = self._config
            self._config = config
        return previous

    def update(self, **changes):
        with self._lock:
            previous = self._config
            self._config = previous.with_updates(**changes)
            return previous, self._config


def _normalize_roi_tuple(roi):
    if roi is None:
        return None
    try:
        x, y, width, height = [int(value) for value in roi]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def playback_restart_required(previous, current, input_path):
    if previous.nn_interval_ms != current.nn_interval_ms:
        return True
    return previous.roi != current.roi and not input_path
