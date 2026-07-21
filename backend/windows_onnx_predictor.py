"""Windows-native ONNX predictor used by the local inference server."""

from pathlib import Path

import numpy as np

from backend.ellipse_decoder import decode_ellipse_vsa
from backend.inference_request import process_inference_request
from backend.model_assets import MODE_CENTER, MODE_ELLIPSE
from backend.protocol import make_error_response
from backend.settings import DEFAULT_SENSOR_HEIGHT, DEFAULT_SENSOR_WIDTH
from backend.windows_onnx_runtime import prepare_windows_cuda_runtime


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
    fps0 = furthest_point_sample_indices(xyz0, 512, rng)
    xyz1 = xyz0[np.sort(fps0)]
    fps1 = furthest_point_sample_indices(xyz1, 256, rng)
    xyz2 = xyz1[np.sort(fps1)]
    fps2 = furthest_point_sample_indices(xyz2, 128, rng)
    return fps0[None], fps1[None], fps2[None]


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
        expected_names = {"events", "fps0", "fps1", "fps2"}
        actual_names = {item.name for item in self.session.get_inputs()}
        if actual_names != expected_names:
            raise RuntimeError(
                f"Unexpected {self.mode_name} model inputs: {sorted(actual_names)}"
            )
        self.rng = np.random.default_rng(seed)
        self.load_message = (
            f"Windows ONNX CUDA {self.mode_name} model loaded\n"
            f"Model: {self.model_path}\n"
            f"Provider: {self.session.get_providers()[0]}"
        )

    def run_model(self, event_data):
        event_data = np.ascontiguousarray(event_data, dtype=np.float32)
        if event_data.shape != (1024, 3):
            raise ValueError(
                f"{self.mode_name} model requires event shape (1024, 3), "
                f"got {event_data.shape}"
            )
        fps0, fps1, fps2 = build_fps_inputs(event_data, self.rng)
        return self.session.run(
            None,
            {
                "events": event_data.T[None],
                "fps0": fps0,
                "fps1": fps1,
                "fps2": fps2,
            },
        )[0]


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
