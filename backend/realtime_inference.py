import logging

import torch

from backend.inference_request import process_inference_request
from backend.model_assets import MODE_CENTER, MODE_ELLIPSE
from backend.predictor_registry import PredictorRegistry, PredictorSpec
from backend.protocol import make_error_response
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH


LOGGER = logging.getLogger(__name__)

_LEGACY_PREDICTOR_EXPORTS = {
    "BasePredictor",
    "CenterPredictor",
    "EllipsePredictor",
    "inplace_relu",
}


def create_default_predictor_registry():
    from backend.eventmamba_predictors import CenterPredictor, EllipsePredictor

    return PredictorRegistry(
        (
            PredictorSpec(MODE_CENTER, CenterPredictor),
            PredictorSpec(MODE_ELLIPSE, EllipsePredictor),
        )
    )


class EventMambaPredictor:
    def __init__(
        self,
        center_weights=None,
        ellipse_weights=None,
        initial_mode=MODE_CENTER,
        predictor_registry=None,
        weights_by_mode=None,
        device=None,
    ):
        self.width = DEFAULT_SENSOR_WIDTH
        self.height = DEFAULT_SENSOR_HEIGHT
        self.center_weights = center_weights
        self.ellipse_weights = ellipse_weights
        self.weights_by_mode = {
            MODE_CENTER: center_weights,
            MODE_ELLIPSE: ellipse_weights,
        }
        if weights_by_mode:
            self.weights_by_mode.update(
                {str(mode).lower(): path for mode, path in weights_by_mode.items()}
            )

        self.registry = predictor_registry or create_default_predictor_registry()
        self.current_mode = str(initial_mode or MODE_CENTER).lower()
        self.current_weights = None
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.predictor = None
        self.load_message = ""
        self._build_predictor(self.current_mode)

    @property
    def available_modes(self):
        return self.registry.modes

    def _build_predictor(self, mode):
        mode = str(mode or "").lower()
        weights_path = self.weights_by_mode.get(mode)
        self.predictor = self.registry.create(mode, weights_path, self.device)
        self.current_mode = mode
        self.current_weights = weights_path
        self.load_message = getattr(self.predictor, "load_message", "")

    def set_mode(self, mode):
        """Switch to the registered predictor for a mode."""
        mode = str(mode or "").lower()
        if not self.registry.supports(mode):
            raise ValueError(f"Unsupported prediction mode: {mode}")

        weights_path = self.weights_by_mode.get(mode)
        if not weights_path:
            LOGGER.warning("No weights configured for %s mode; keeping current mode", mode)
            return False
        if mode == self.current_mode and weights_path == self.current_weights:
            return False

        self._build_predictor(mode)
        LOGGER.info("Prediction mode switched to %s, weights: %s", mode, weights_path)
        return True

    def process_data(self, data):
        try:
            return process_inference_request(self, data)
        except Exception as exc:
            error_message = f"模型推理出错: {exc}"
            LOGGER.exception(error_message)
            return make_error_response(error_message)


def __getattr__(name):
    if name not in _LEGACY_PREDICTOR_EXPORTS:
        raise AttributeError(name)
    from backend import eventmamba_predictors

    return getattr(eventmamba_predictors, name)
