import pytest

from backend.predictor_registry import PredictorRegistry, PredictorSpec


class FakePredictor:
    def __init__(self, weights_path, device):
        self.weights_path = weights_path
        self.device = device


def test_registry_normalizes_mode_and_builds_predictor():
    registry = PredictorRegistry((PredictorSpec(" Center ", FakePredictor),))

    predictor = registry.create("CENTER", "weights.pth", "cpu")

    assert registry.modes == ("center",)
    assert registry.supports("Center")
    assert predictor.weights_path == "weights.pth"
    assert predictor.device == "cpu"


def test_registry_rejects_duplicate_mode_unless_replaced():
    registry = PredictorRegistry((PredictorSpec("center", FakePredictor),))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(PredictorSpec("center", lambda *_: object()))

    replacement = PredictorSpec("center", lambda *_: "replacement")
    registry.register(replacement, replace=True)

    assert registry.create("center", None, None) == "replacement"


def test_registry_reports_unsupported_mode():
    registry = PredictorRegistry((PredictorSpec("center", FakePredictor),))

    with pytest.raises(ValueError, match="Unsupported prediction mode: ellipse"):
        registry.create("ellipse", "weights.pth", "cpu")
