from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QByteArray, QSettings

from backend.event_processing import normalize_noise_filter_type
from backend.settings import DEFAULT_FPS, DEFAULT_NOISE_FILTER_THRESHOLD_US


SUPPORTED_PALETTES = frozenset(("Dark", "Light", "CoolWarm", "Gray"))
SUPPORTED_REPLAY_SPEEDS = frozenset(("0.25x", "0.5x", "1x", "2x", "4x"))
SUPPORTED_PREDICTION_MODES = frozenset(("center", "ellipse"))


@dataclass(frozen=True)
class UiPreferences:
    palette: str = "Dark"
    fps: float = DEFAULT_FPS
    replay_speed: str = "1x"
    prediction_mode: str = "center"
    noise_filter_type: str = "none"
    noise_filter_threshold_us: int = DEFAULT_NOISE_FILTER_THRESHOLD_US
    log_collapsed: bool = False
    settings_panel_visible: bool = False


class PreferenceStore:
    def __init__(self, settings=None):
        self.settings = settings or QSettings("UI_Event", "UI_Event")

    def load(self):
        palette = str(self.settings.value("capture/palette", "Dark"))
        if palette not in SUPPORTED_PALETTES:
            palette = "Dark"
        replay_speed = str(self.settings.value("capture/replay_speed", "1x"))
        if replay_speed not in SUPPORTED_REPLAY_SPEEDS:
            replay_speed = "1x"
        prediction_mode = str(self.settings.value("inference/prediction_mode", "center"))
        if prediction_mode not in SUPPORTED_PREDICTION_MODES:
            prediction_mode = "center"
        return UiPreferences(
            palette=palette,
            fps=_bounded_float(
                self.settings.value("capture/fps", DEFAULT_FPS),
                default=DEFAULT_FPS,
                minimum=1.0,
                maximum=999.99,
            ),
            replay_speed=replay_speed,
            prediction_mode=prediction_mode,
            noise_filter_type=normalize_noise_filter_type(
                self.settings.value("processing/noise_filter", "none")
            ),
            noise_filter_threshold_us=_bounded_int(
                self.settings.value(
                    "processing/noise_threshold_us",
                    DEFAULT_NOISE_FILTER_THRESHOLD_US,
                ),
                default=DEFAULT_NOISE_FILTER_THRESHOLD_US,
                minimum=1,
                maximum=1_000_000,
            ),
            log_collapsed=_as_bool(
                self.settings.value("window/log_collapsed", False)
            ),
            settings_panel_visible=_as_bool(
                self.settings.value("window/settings_panel_visible", False)
            ),
        )

    def save(self, preferences):
        self.settings.setValue("capture/palette", preferences.palette)
        self.settings.setValue("capture/fps", float(preferences.fps))
        self.settings.setValue("capture/replay_speed", preferences.replay_speed)
        self.settings.setValue(
            "inference/prediction_mode",
            preferences.prediction_mode,
        )
        self.settings.setValue(
            "processing/noise_filter",
            preferences.noise_filter_type,
        )
        self.settings.setValue(
            "processing/noise_threshold_us",
            int(preferences.noise_filter_threshold_us),
        )
        self.settings.setValue("window/log_collapsed", preferences.log_collapsed)
        self.settings.setValue(
            "window/settings_panel_visible",
            preferences.settings_panel_visible,
        )
        self.settings.sync()

    def restore_window_geometry(self, window):
        geometry = self.settings.value("window/geometry")
        if geometry is None:
            return False
        if isinstance(geometry, bytes):
            geometry = QByteArray(geometry)
        try:
            return bool(window.restoreGeometry(geometry))
        except (TypeError, RuntimeError):
            return False

    def save_window_geometry(self, window):
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.sync()

    def last_input_directory(self, fallback):
        return _existing_directory(
            self.settings.value("directories/input"),
            fallback,
        )

    def last_model_directory(self, fallback):
        return _existing_directory(
            self.settings.value("directories/model"),
            fallback,
        )

    def remember_input_file(self, file_path):
        self._remember_parent("directories/input", file_path)

    def remember_model_file(self, file_path):
        self._remember_parent("directories/model", file_path)

    def _remember_parent(self, key, file_path):
        if not file_path:
            return
        self.settings.setValue(key, str(Path(file_path).resolve().parent))
        self.settings.sync()


def _bounded_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(float(minimum), min(float(maximum), value))


def _bounded_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(int(minimum), min(int(maximum), value))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _existing_directory(value, fallback):
    if value:
        directory = Path(str(value))
        if directory.is_dir():
            return str(directory)
    return str(fallback)
