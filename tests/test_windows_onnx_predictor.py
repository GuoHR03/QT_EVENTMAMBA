import numpy as np

from backend.windows_onnx_predictor import (
    WindowsOnnxCenterPredictor,
    WindowsOnnxEllipsePredictor,
    WindowsOnnxPredictorRuntime,
    build_fps_inputs,
    furthest_point_sample_indices,
)


def test_furthest_point_sample_indices_are_valid_and_deterministic():
    points = np.random.default_rng(3).standard_normal((32, 3), dtype=np.float32)

    first = furthest_point_sample_indices(points, 12, np.random.default_rng(7))
    second = furthest_point_sample_indices(points, 12, np.random.default_rng(7))

    assert first.dtype == np.int64
    assert first.shape == (12,)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first < len(points)))


def test_build_fps_inputs_matches_center_model_shapes():
    events = np.random.default_rng(4).standard_normal((1024, 3), dtype=np.float32)

    fps0, fps1, fps2 = build_fps_inputs(events, np.random.default_rng(7))

    assert fps0.shape == (1, 512)
    assert fps1.shape == (1, 256)
    assert fps2.shape == (1, 128)


def test_center_predictor_builds_onnx_inputs_without_importing_torch():
    class FakeSession:
        def run(self, output_names, inputs):
            assert output_names is None
            assert inputs["events"].shape == (1, 3, 1024)
            assert inputs["fps0"].shape == (1, 512)
            assert inputs["fps1"].shape == (1, 256)
            assert inputs["fps2"].shape == (1, 128)
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

    predictor = object.__new__(WindowsOnnxCenterPredictor)
    predictor.session = FakeSession()
    predictor.rng = np.random.default_rng(7)
    events = np.random.default_rng(5).standard_normal((1024, 3), dtype=np.float32)

    assert predictor.predict(events) == [0.25, 0.75]


def test_ellipse_predictor_decodes_onnx_vector(monkeypatch):
    predictor = object.__new__(WindowsOnnxEllipsePredictor)
    predictor.matrix_a = np.ones((2, 2), dtype=np.float32)
    monkeypatch.setattr(
        predictor,
        "run_model",
        lambda _events: np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )

    result = predictor.predict(np.zeros((1024, 3), dtype=np.float32))

    assert len(result) == 5


def test_windows_runtime_switches_to_ellipse_predictor():
    runtime = object.__new__(WindowsOnnxPredictorRuntime)
    center = object()
    ellipse = object()
    runtime.predictors = {"center": center, "ellipse": ellipse}
    runtime.current_mode = "center"
    runtime.predictor = center

    assert runtime.set_mode("ellipse") is True
    assert runtime.current_mode == "ellipse"
    assert runtime.predictor is ellipse
