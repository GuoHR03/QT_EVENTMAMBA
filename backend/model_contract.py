"""Shared model capabilities used across UI, transport, and runtimes.

Keep structural model facts here so adding a prediction mode does not require
duplicating validation rules in every layer. Runtime-specific asset paths and
predictor factories intentionally remain in their owning modules.
"""

from dataclasses import dataclass
from types import MappingProxyType


MODE_CENTER = "center"
MODE_ELLIPSE = "ellipse"

EVENT_POINT_COUNT = 1024
EVENT_FEATURE_COUNT = 3
EVENT_SHAPE = (EVENT_POINT_COUNT, EVENT_FEATURE_COUNT)
MODEL_INPUT_SHAPE = (1, EVENT_FEATURE_COUNT, EVENT_POINT_COUNT)
FPS_STAGE_COUNTS = (512, 256, 128)


@dataclass(frozen=True)
class ModelSpec:
    """Stable cross-layer contract for one prediction mode."""

    mode: str
    output_size: int
    requires_matrix: bool = False

    def __post_init__(self):
        normalized_mode = str(self.mode or "").strip().lower()
        if not normalized_mode:
            raise ValueError("model mode is required")
        if int(self.output_size) <= 0:
            raise ValueError("model output_size must be positive")
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "output_size", int(self.output_size))


MODEL_SPECS = MappingProxyType(
    {
        MODE_CENTER: ModelSpec(MODE_CENTER, output_size=2),
        MODE_ELLIPSE: ModelSpec(
            MODE_ELLIPSE,
            output_size=5,
            requires_matrix=True,
        ),
    }
)
SUPPORTED_MODEL_MODES = tuple(MODEL_SPECS)


def normalize_model_mode(mode):
    return str(mode or "").strip().lower()


def get_model_spec(mode):
    normalized_mode = normalize_model_mode(mode)
    try:
        return MODEL_SPECS[normalized_mode]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_MODEL_MODES)
        raise ValueError(
            f"Unsupported prediction mode: {normalized_mode}; supported: {supported}"
        ) from exc


def is_supported_model_mode(mode):
    return normalize_model_mode(mode) in MODEL_SPECS
