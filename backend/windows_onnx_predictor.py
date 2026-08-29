"""Windows-native ONNX predictor used by the local inference server."""

from pathlib import Path

import numpy as np

from backend.ellipse_decoder import decode_ellipse_vsa
from backend.inference_request import process_inference_request
from backend.model_assets import MODE_CENTER, MODE_ELLIPSE
from backend.model_contract import (
    EVENT_SHAPE,
    FPS_STAGE_COUNTS,
    MODEL_INPUT_SHAPE,
)
from backend.protocol import make_error_response
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH
from backend.windows_onnx_runtime import prepare_windows_cuda_runtime


FPS_CONTRACT_NATIVE = "native"
FPS_CONTRACT_LEGACY = "legacy"

_NATIVE_INPUT_SPEC = {
    "events": (MODEL_INPUT_SHAPE, "tensor(float)"),
    "fps_starts": ((1, 3), "tensor(int64)"),
}
_LEGACY_INPUT_SPEC = {
    "events": (MODEL_INPUT_SHAPE, "tensor(float)"),
    "fps0": ((1, FPS_STAGE_COUNTS[0]), "tensor(int64)"),
    "fps1": ((1, FPS_STAGE_COUNTS[1]), "tensor(int64)"),
    "fps2": ((1, FPS_STAGE_COUNTS[2]), "tensor(int64)"),
}


def _model_input_contract(model_inputs, mode_name):
    input_by_name = {item.name: item for item in model_inputs}
    actual_names = set(input_by_name)
    if len(input_by_name) != len(model_inputs):
        raise RuntimeError(f"Duplicate {mode_name} model input names")

    if actual_names == set(_NATIVE_INPUT_SPEC):
        contract = FPS_CONTRACT_NATIVE
        expected_spec = _NATIVE_INPUT_SPEC
    elif actual_names == set(_LEGACY_INPUT_SPEC):
        contract = FPS_CONTRACT_LEGACY
        expected_spec = _LEGACY_INPUT_SPEC
    else:
        raise RuntimeError(
            f"Unexpected {mode_name} model inputs: {sorted(actual_names)}; "
            f"expected {sorted(_NATIVE_INPUT_SPEC)} or "
            f"{sorted(_LEGACY_INPUT_SPEC)}"
        )

    for name, (expected_shape, expected_type) in expected_spec.items():
        item = input_by_name[name]
        actual_shape = tuple(item.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {mode_name} model input shape for {name}: "
                f"{actual_shape}; expected {expected_shape}"
            )
        if item.type != expected_type:
            raise RuntimeError(
                f"Unexpected {mode_name} model input type for {name}: "
                f"{item.type}; expected {expected_type}"
            )
    return contract


def furthest_point_sample_indices(points, count, rng):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"FPS points must be two-dimensional, got {points.shape}")
    if count <= 0 or count > len(points):
        raise ValueError(f"FPS count {count} is invalid for {len(points)} points")

    centroids = np.zeros(count, dtype=np.int64)
    distance = np.full(len(points), np.float32(1e10), dtype=np.float32)
    farthest = int(rng.integers(0, len(points)))
    for index in range(count):
        centroids[index] = farthest
        delta = points - points[farthest]
        distance = np.minimum(distance, np.sum(delta * delta, axis=1))
        farthest = int(np.argmax(distance))
    return centroids


def build_fps_inputs(events, rng):
    xyz0 = np.asarray(events, dtype=np.float32)
    fps0 = furthest_point_sample_indices(xyz0, FPS_STAGE_COUNTS[0], rng)
    xyz1 = xyz0[np.sort(fps0)]
    fps1 = furthest_point_sample_indices(xyz1, FPS_STAGE_COUNTS[1], rng)
    xyz2 = xyz1[np.sort(fps1)]
    fps2 = furthest_point_sample_indices(xyz2, FPS_STAGE_COUNTS[2], rng)
    return fps0[None], fps1[None], fps2[None]


def build_fps_starts(rng):
    """Return the three RNG starts consumed by hierarchical native FPS."""
    starts = np.empty((1, 3), dtype=np.int64)
    starts[0, 0] = rng.integers(0, EVENT_SHAPE[0])
    starts[0, 1] = rng.integers(0, FPS_STAGE_COUNTS[0])
    starts[0, 2] = rng.integers(0, FPS_STAGE_COUNTS[1])
    return starts


class WindowsOnnxPredictor:
    mode_name = "model"

    def __init__(self, model_path, custom_op_library, seed=7):
        import onnxruntime as ort

        self.model_path = str(Path(model_path).resolve())
        self.custom_op_library = str(Path(custom_op_library).resolve())
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"Windows ONNX model not found: {self.model_path}")
        if not Path(self.custom_op_library).is_file():
            raise FileNotFoundError(
                f"Selective scan CUDA library not found: {self.custom_op_library}"
            )

        self.runtime_directories = prepare_windows_cuda_runtime()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                f"CUDAExecutionProvider is unavailable: {ort.get_available_providers()}"
            )

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.register_custom_ops_library(self.custom_op_library)
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"ONNX Runtime fell back to {self.session.get_providers()[0]}"
            )
        self.fps_contract = _model_input_contract(
            self.session.get_inputs(),
            self.mode_name,
        )
        self.rng = np.random.default_rng(seed)
        self.load_message = (
            f"Windows ONNX CUDA {self.mode_name} model loaded\n"
            f"Model: {self.model_path}\n"
            f"Provider: {self.session.get_providers()[0]}\n"
            f"FPS: {self.fps_contract}"
        )

    def run_model(self, event_data):
        event_data = np.ascontiguousarray(event_data, dtype=np.float32)
        if event_data.shape != EVENT_SHAPE:
            raise ValueError(
                f"{self.mode_name} model requires event shape {EVENT_SHAPE}, "
                f"got {event_data.shape}"
            )
        inputs = {
            "events": np.ascontiguousarray(event_data.T[None], dtype=np.float32),
        }
        if self.fps_contract == FPS_CONTRACT_NATIVE:
            inputs["fps_starts"] = build_fps_starts(self.rng)
        elif self.fps_contract == FPS_CONTRACT_LEGACY:
            fps0, fps1, fps2 = build_fps_inputs(event_data, self.rng)
            inputs.update(
                {
                    "fps0": fps0,
                    "fps1": fps1,
                    "fps2": fps2,
                }
            )
        else:
            raise RuntimeError(f"Unsupported FPS input contract: {self.fps_contract}")
        return self.session.run(None, inputs)[0]


class WindowsOnnxCenterPredictor(WindowsOnnxPredictor):
    mode_name = MODE_CENTER

    def predict(self, event_data):
        output = self.run_model(event_data)
        return np.asarray(output).reshape(-1).tolist()


class WindowsOnnxEllipsePredictor(WindowsOnnxPredictor):
    mode_name = MODE_ELLIPSE

    def __init__(self, model_path, matrix_path, custom_op_library, seed=7):
        self.matrix_path = str(Path(matrix_path).resolve())
        if not Path(self.matrix_path).is_file():
            raise FileNotFoundError(
                f"Ellipse matrix_A file not found: {self.matrix_path}"
            )
        self.matrix_a = np.load(self.matrix_path)
        super().__init__(model_path, custom_op_library, seed=seed)
        self.load_message += f"\nMatrix: {self.matrix_path}"

    def predict(self, event_data):
        output = self.run_model(event_data)
        decoded = decode_ellipse_vsa(output, self.matrix_a)
        return decoded.reshape(-1).tolist()


class WindowsOnnxPredictorRuntime:
    def __init__(
        self,
        center_model,
        ellipse_model,
        ellipse_matrix,
        custom_op_library,
        initial_mode=MODE_CENTER,
    ):
        self.width = DEFAULT_SENSOR_WIDTH
        self.height = DEFAULT_SENSOR_HEIGHT
        self.predictors = {
            MODE_CENTER: WindowsOnnxCenterPredictor(
                center_model,
                custom_op_library,
            ),
            MODE_ELLIPSE: WindowsOnnxEllipsePredictor(
                ellipse_model,
                ellipse_matrix,
                custom_op_library,
            ),
        }
        self.current_mode = None
        self.predictor = None
        self.set_mode(initial_mode)
        self.load_message = "\n".join(
            predictor.load_message for predictor in self.predictors.values()
        )

    def set_mode(self, mode):
        normalized = str(mode or "").strip().lower()
        if normalized not in self.predictors:
            raise ValueError(f"Unsupported Windows ONNX mode: {normalized}")
        if normalized == self.current_mode:
            return False
        self.current_mode = normalized
        self.predictor = self.predictors[normalized]
        return True

    def process_data(self, data):
        try:
            return process_inference_request(self, data)
        except Exception as exc:
            return make_error_response(f"Windows ONNX inference failed: {exc}")
