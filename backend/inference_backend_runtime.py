"""Runtime-specific inference backend launch construction."""

import os
import math
from dataclasses import dataclass

from backend.settings import (
    DEFAULT_BACKEND_READY_TIMEOUT_S,
    DEFAULT_CENTER_ONNX_MODEL,
    DEFAULT_ELLIPSE_MATRIX,
    DEFAULT_ELLIPSE_ONNX_MODEL,
    DEFAULT_INFERENCE_RUNTIME,
    DEFAULT_LINUX_PYTHON,
    DEFAULT_SELECTIVE_SCAN_DLL,
    DEFAULT_WINDOWS_BACKEND_EXECUTABLE,
    DEFAULT_WINDOWS_PYTHON,
    DEFAULT_WSL_DISTRO,
    ENV_BACKEND_READY_TIMEOUT_S,
    ENV_CENTER_ONNX_MODEL,
    ENV_ELLIPSE_MATRIX,
    ENV_ELLIPSE_ONNX_MODEL,
    ENV_INFERENCE_RUNTIME,
    ENV_LINUX_PYTHON,
    ENV_SELECTIVE_SCAN_DLL,
    ENV_WINDOWS_BACKEND_EXECUTABLE,
    ENV_WINDOWS_PYTHON,
    ENV_WSL_DISTRO,
)
from backend.windows_process import (
    WindowsBackendProcess,
    build_windows_backend_command,
    build_windows_backend_executable_command,
)
from backend.wsl_process import WslBackendProcess, build_backend_command


@dataclass(frozen=True)
class BackendLaunch:
    command: list
    active_model_path: str


@dataclass(frozen=True)
class InferenceRuntimeSettings:
    kind: str
    wsl_distro: str
    linux_python: str
    windows_python: str
    windows_backend_executable: str
    center_onnx_model: str
    ellipse_onnx_model: str
    ellipse_matrix: str
    selective_scan_dll: str
    ready_timeout_s: float

    @classmethod
    def from_environment(cls, environ=None):
        environ = os.environ if environ is None else environ
        kind = environ.get(ENV_INFERENCE_RUNTIME, DEFAULT_INFERENCE_RUNTIME).strip().lower()
        if kind not in ("windows", "wsl"):
            raise ValueError(
                f"Unsupported inference runtime: {kind}; use windows or wsl"
            )
        return cls(
            kind=kind,
            wsl_distro=environ.get(ENV_WSL_DISTRO, DEFAULT_WSL_DISTRO),
            linux_python=environ.get(ENV_LINUX_PYTHON, DEFAULT_LINUX_PYTHON),
            windows_python=environ.get(ENV_WINDOWS_PYTHON, DEFAULT_WINDOWS_PYTHON),
            windows_backend_executable=environ.get(
                ENV_WINDOWS_BACKEND_EXECUTABLE,
                DEFAULT_WINDOWS_BACKEND_EXECUTABLE,
            ),
            center_onnx_model=environ.get(
                ENV_CENTER_ONNX_MODEL,
                DEFAULT_CENTER_ONNX_MODEL,
            ),
            ellipse_onnx_model=environ.get(
                ENV_ELLIPSE_ONNX_MODEL,
                DEFAULT_ELLIPSE_ONNX_MODEL,
            ),
            ellipse_matrix=environ.get(ENV_ELLIPSE_MATRIX, DEFAULT_ELLIPSE_MATRIX),
            selective_scan_dll=environ.get(
                ENV_SELECTIVE_SCAN_DLL,
                DEFAULT_SELECTIVE_SCAN_DLL,
            ),
            ready_timeout_s=finite_positive_timeout(
                environ.get(
                    ENV_BACKEND_READY_TIMEOUT_S,
                    DEFAULT_BACKEND_READY_TIMEOUT_S,
                ),
                DEFAULT_BACKEND_READY_TIMEOUT_S,
            ),
        )

    @property
    def display_name(self):
        return "Windows ONNX CUDA" if self.kind == "windows" else "WSL"

    def create_process(self, log_path):
        if self.kind == "windows":
            return WindowsBackendProcess(log_path)
        return WslBackendProcess(self.wsl_distro, log_path)


def finite_positive_timeout(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value <= 0:
        return float(default)
    return value


def resolve_runtime_path(path, project_dir):
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if not os.path.isabs(expanded):
        expanded = os.path.join(project_dir, expanded)
    return os.path.abspath(expanded)


def build_windows_launch(
    project_dir,
    weights_path,
    prediction_mode,
    port,
    instance_nonce,
    *,
    center_model,
    ellipse_model,
    ellipse_matrix,
    custom_op_library,
    python_executable,
    backend_executable,
    frozen,
    is_file=os.path.isfile,
):
    center_model_path = resolve_runtime_path(center_model, project_dir)
    ellipse_model_path = resolve_runtime_path(ellipse_model, project_dir)
    selected_path = str(weights_path)
    selected_name = os.path.basename(selected_path).lower()
    if selected_path.lower().endswith(".onnx"):
        selected_path = resolve_runtime_path(selected_path, project_dir)
        if "ellipse" in selected_name:
            ellipse_model_path = selected_path
        elif "center" in selected_name:
            center_model_path = selected_path
        elif prediction_mode == "ellipse":
            ellipse_model_path = selected_path
        else:
            center_model_path = selected_path

    ellipse_matrix_path = resolve_runtime_path(ellipse_matrix, project_dir)
    custom_op_path = resolve_runtime_path(custom_op_library, project_dir)
    required = {
        "center ONNX model": center_model_path,
        "ellipse ONNX model": ellipse_model_path,
        "ellipse matrix_A": ellipse_matrix_path,
        "selective scan CUDA DLL": custom_op_path,
    }
    if frozen:
        executable_path = resolve_runtime_path(backend_executable, project_dir)
        required = {"Windows backend executable": executable_path, **required}
    else:
        python_path = resolve_runtime_path(python_executable, project_dir)
        backend_script = os.path.join(project_dir, "windows_backend.py")
        required = {
            "Windows Python": python_path,
            "Windows backend script": backend_script,
            **required,
        }

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not is_file(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Windows ONNX inference files are missing:\n" + "\n".join(missing)
        )

    active_model_path = (
        ellipse_model_path if prediction_mode == "ellipse" else center_model_path
    )
    if frozen:
        command = build_windows_backend_executable_command(
            executable_path,
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_path,
            prediction_mode,
            port,
            instance_nonce,
        )
    else:
        command = build_windows_backend_command(
            python_path,
            backend_script,
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_path,
            prediction_mode,
            port,
            instance_nonce,
        )
    return BackendLaunch(command, active_model_path)


def build_wsl_launch(
    project_dir,
    weights_path,
    prediction_mode,
    port,
    instance_nonce,
    *,
    distro,
    linux_python,
    path_converter,
):
    linux_script = os.path.join(project_dir, "linux_backend.py")
    command = build_backend_command(
        distro,
        linux_python,
        path_converter(linux_script),
        path_converter(weights_path),
        prediction_mode,
        port,
        instance_nonce,
    )
    return BackendLaunch(command, str(weights_path))
