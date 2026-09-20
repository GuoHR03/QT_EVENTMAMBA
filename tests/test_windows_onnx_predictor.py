import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend.windows_onnx_predictor import (
    FPS_CONTRACT_LEGACY,
    FPS_CONTRACT_NATIVE,
    FPS_CONTRACT_RANDLA,
    WindowsOnnxCenterPredictor,
    WindowsOnnxEllipsePredictor,
    WindowsOnnxPredictorRuntime,
    _model_input_contract,
    build_fps_inputs,
    build_fps_starts,
    build_random_sample_inputs,
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


def test_build_fps_starts_preserves_existing_rng_sequence():
    starts = build_fps_starts(np.random.default_rng(7))

    assert starts.tolist() == [[967, 320, 175]]
    assert starts.dtype == np.int64
    assert starts.flags.c_contiguous


def test_build_fps_starts_match_legacy_stage_start_indices():
    events = np.random.default_rng(4).standard_normal((1024, 3), dtype=np.float32)
    fps0, fps1, fps2 = build_fps_inputs(events, np.random.default_rng(7))

    starts = build_fps_starts(np.random.default_rng(7))

    assert starts.tolist() == [[fps0[0, 0], fps1[0, 0], fps2[0, 0]]]


def test_build_random_sample_inputs_are_unique_sorted_and_deterministic():
    first = build_random_sample_inputs(np.random.default_rng(2026))
    second = build_random_sample_inputs(np.random.default_rng(2026))

    for actual, repeated, population, count in zip(
        first,
        second,
        (1024, 512, 256),
        (512, 256, 128),
    ):
        assert actual.shape == (1, count)
        assert actual.dtype == np.int64
        assert actual.flags.c_contiguous
        assert np.array_equal(actual, repeated)
        assert len(np.unique(actual)) == count
        assert np.all(actual[:, 1:] > actual[:, :-1])
        assert np.all((actual >= 0) & (actual < population))


class FakeModelInput:
    def __init__(self, name, shape, input_type):
        self.name = name
        self.shape = shape
        self.type = input_type


def _model_inputs(spec):
    return [FakeModelInput(name, shape, input_type) for name, shape, input_type in spec]


def test_model_input_contract_accepts_native_and_legacy_shapes():
    native = _model_inputs(
        [
            ("events", [1, 3, 1024], "tensor(float)"),
            ("fps_starts", [1, 3], "tensor(int64)"),
        ]
    )
    legacy = _model_inputs(
        [
            ("events", [1, 3, 1024], "tensor(float)"),
            ("fps0", [1, 512], "tensor(int64)"),
            ("fps1", [1, 256], "tensor(int64)"),
            ("fps2", [1, 128], "tensor(int64)"),
        ]
    )
    randla = _model_inputs(
        [
            ("events", [1, 3, 1024], "tensor(float)"),
            ("sample0", [1, 512], "tensor(int64)"),
            ("sample1", [1, 256], "tensor(int64)"),
            ("sample2", [1, 128], "tensor(int64)"),
        ]
    )

    assert _model_input_contract(native, "center") == FPS_CONTRACT_NATIVE
    assert _model_input_contract(legacy, "center") == FPS_CONTRACT_LEGACY
    assert _model_input_contract(randla, "ellipse") == FPS_CONTRACT_RANDLA


@pytest.mark.parametrize(
    "inputs, message",
    [
        (
            _model_inputs(
                [
                    ("events", [1, 3, 1024], "tensor(float)"),
                    ("fps_starts", [1, 3], "tensor(int64)"),
                    ("fps0", [1, 512], "tensor(int64)"),
                ]
            ),
            "Unexpected center model inputs",
        ),
        (
            _model_inputs(
                [
                    ("events", [None, 3, 1024], "tensor(float)"),
                    ("fps_starts", [1, 3], "tensor(int64)"),
                ]
            ),
            "input shape for events",
        ),
        (
            _model_inputs(
                [
                    ("events", [1, 3, 1024], "tensor(double)"),
                    ("fps_starts", [1, 3], "tensor(int64)"),
                ]
            ),
            "input type for events",
        ),
    ],
)
def test_model_input_contract_rejects_mixed_dynamic_or_wrong_type(inputs, message):
    with pytest.raises(RuntimeError, match=message):
        _model_input_contract(inputs, "center")


def test_center_predictor_builds_legacy_onnx_inputs_without_importing_torch():
    class FakeSession:
        def run(self, output_names, inputs):
            assert output_names is None
            assert inputs["events"].shape == (1, 3, 1024)
            assert inputs["events"].dtype == np.float32
            assert inputs["events"].flags.c_contiguous
            assert inputs["fps0"].shape == (1, 512)
            assert inputs["fps1"].shape == (1, 256)
            assert inputs["fps2"].shape == (1, 128)
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

    predictor = object.__new__(WindowsOnnxCenterPredictor)
    predictor.session = FakeSession()
    predictor.rng = np.random.default_rng(7)
    predictor.fps_contract = FPS_CONTRACT_LEGACY
    events = np.asfortranarray(
        np.random.default_rng(5).standard_normal((1024, 3))
    )

    assert predictor.predict(events) == [0.25, 0.75]


def test_center_predictor_native_contract_never_runs_python_fps(monkeypatch):
    class FakeSession:
        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"events", "fps_starts"}
            assert inputs["events"].shape == (1, 3, 1024)
            assert inputs["events"].dtype == np.float32
            assert inputs["events"].flags.c_contiguous
            assert inputs["fps_starts"].tolist() == [[967, 320, 175]]
            assert inputs["fps_starts"].dtype == np.int64
            assert inputs["fps_starts"].flags.c_contiguous
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

    def fail_if_python_fps_runs(*_args, **_kwargs):
        raise AssertionError("native contract must not execute Python FPS")

    monkeypatch.setattr(
        "backend.windows_onnx_predictor.build_fps_inputs",
        fail_if_python_fps_runs,
    )
    predictor = object.__new__(WindowsOnnxCenterPredictor)
    predictor.session = FakeSession()
    predictor.rng = np.random.default_rng(7)
    predictor.fps_contract = FPS_CONTRACT_NATIVE
    events = np.asfortranarray(
        np.random.default_rng(5).standard_normal((1024, 3))
    )

    assert predictor.predict(events) == [0.25, 0.75]


def test_predictor_builds_randla_random_sample_inputs():
    class FakeSession:
        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"events", "sample0", "sample1", "sample2"}
            assert inputs["sample0"].shape == (1, 512)
            assert inputs["sample1"].shape == (1, 256)
            assert inputs["sample2"].shape == (1, 128)
            return [np.zeros((1, 1024), dtype=np.float32)]

    predictor = object.__new__(WindowsOnnxEllipsePredictor)
    predictor.session = FakeSession()
    predictor.rng = np.random.default_rng(2026)
    predictor.fps_contract = FPS_CONTRACT_RANDLA

    output = predictor.run_model(np.zeros((1024, 3), dtype=np.float32))

    assert output.shape == (1, 1024)


def test_native_predictor_warms_up_without_consuming_live_rng(
    monkeypatch,
    tmp_path,
):
    runs = []

    class FakeSessionOptions:
        def __init__(self):
            self.log_severity_level = None

        def register_custom_ops_library(self, _path):
            return None

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def get_inputs(self):
            return _model_inputs(
                [
                    ("events", [1, 3, 1024], "tensor(float)"),
                    ("fps_starts", [1, 3], "tensor(int64)"),
                ]
            )

        def run(self, _output_names, inputs):
            runs.append({name: value.copy() for name, value in inputs.items()})
            return [np.zeros((1, 2), dtype=np.float32)]

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
        get_available_providers=lambda: ["CUDAExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        "backend.windows_onnx_predictor.prepare_windows_cuda_runtime",
        lambda: (),
    )
    model_path = tmp_path / "center.onnx"
    library_path = tmp_path / "custom.dll"
    model_path.write_bytes(b"model")
    library_path.write_bytes(b"library")

    predictor = WindowsOnnxCenterPredictor(model_path, library_path, seed=7)

    assert len(runs) == 1
    assert "Warmup: complete" in predictor.load_message
    predictor.predict(np.zeros((1024, 3), dtype=np.float32))
    assert len(runs) == 2
    assert runs[0]["fps_starts"].tolist() == [[967, 320, 175]]
    assert runs[1]["fps_starts"].tolist() == [[967, 320, 175]]


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
