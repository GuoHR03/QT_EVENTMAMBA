from backend.predictor_registry import PredictorRegistry, PredictorSpec
from backend.realtime_inference import EventMambaPredictor


class FakePredictor:
    def __init__(self, mode, weights_path, device, calls):
        self.mode = mode
        self.weights_path = weights_path
        self.device = device
        self.load_message = f"loaded {mode}"
        calls.append((mode, weights_path, device))

    def predict(self, event_data):
        return [1.0, 2.0]


def _registry(calls):
    return PredictorRegistry(
        (
            PredictorSpec(
                "center",
                lambda weights, device: FakePredictor(
                    "center", weights, device, calls
                ),
            ),
            PredictorSpec(
                "ellipse",
                lambda weights, device: FakePredictor(
                    "ellipse", weights, device, calls
                ),
            ),
        )
    )


def test_eventmamba_predictor_uses_registry_and_switches_modes():
    calls = []
    predictor = EventMambaPredictor(
        center_weights="center.pth",
        ellipse_weights="ellipse.pth",
        predictor_registry=_registry(calls),
        device="test-device",
    )

    changed = predictor.set_mode("ellipse")

    assert calls == [
        ("center", "center.pth", "test-device"),
        ("ellipse", "ellipse.pth", "test-device"),
    ]
    assert changed is True
    assert predictor.current_mode == "ellipse"
    assert predictor.current_weights == "ellipse.pth"
    assert predictor.load_message == "loaded ellipse"
    assert predictor.available_modes == ("center", "ellipse")


def test_eventmamba_predictor_keeps_mode_when_target_weights_are_missing():
    calls = []
    predictor = EventMambaPredictor(
        center_weights="center.pth",
        predictor_registry=_registry(calls),
        device="cpu",
    )

    changed = predictor.set_mode("ellipse")

    assert changed is False
    assert predictor.current_mode == "center"
    assert calls == [("center", "center.pth", "cpu")]


def test_eventmamba_predictor_accepts_custom_registered_architecture():
    calls = []
    registry = PredictorRegistry(
        (
            PredictorSpec(
                "custom",
                lambda weights, device: FakePredictor(
                    "custom", weights, device, calls
                ),
            ),
        )
    )

    predictor = EventMambaPredictor(
        initial_mode="custom",
        predictor_registry=registry,
        weights_by_mode={"custom": "custom.pth"},
        device="cpu",
    )

    assert predictor.current_mode == "custom"
    assert predictor.current_weights == "custom.pth"
    assert calls == [("custom", "custom.pth", "cpu")]
