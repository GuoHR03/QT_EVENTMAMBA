import os
import sys

from backend.backend_healthcheck import wait_for_backend_ready_with_log
from backend.inference_runtime import default_backend_log_path, read_float_env, runtime_root_dir, to_wsl_path
from backend.settings import (
    DEFAULT_BACKEND_READY_TIMEOUT_S,
    DEFAULT_CENTER_ONNX_MODEL,
    DEFAULT_ELLIPSE_MATRIX,
    DEFAULT_ELLIPSE_ONNX_MODEL,
    DEFAULT_INFERENCE_HOST,
    DEFAULT_INFERENCE_PORT,
    DEFAULT_INFERENCE_RUNTIME,
    DEFAULT_LINUX_PYTHON,
    DEFAULT_NETWORK_TIMEOUT_MS,
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


class InferenceService:
    def __init__(self, frame_queue, prediction_signal):
        self.frame_queue = frame_queue
        self.prediction_signal = prediction_signal
        self.network_thread = None
        self.weights_path = None
        self.prediction_mode = "center"
        self.active_model_path = None
        self.runtime_kind = os.environ.get(
            ENV_INFERENCE_RUNTIME, DEFAULT_INFERENCE_RUNTIME
        ).strip().lower()
        if self.runtime_kind not in ("windows", "wsl"):
            raise ValueError(
                f"Unsupported inference runtime: {self.runtime_kind}; use windows or wsl"
            )
        self.wsl_distro = os.environ.get(ENV_WSL_DISTRO, DEFAULT_WSL_DISTRO)
        self.linux_python = os.environ.get(ENV_LINUX_PYTHON, DEFAULT_LINUX_PYTHON)
        self.windows_python = os.environ.get(
            ENV_WINDOWS_PYTHON, DEFAULT_WINDOWS_PYTHON
        )
        self.windows_backend_executable = os.environ.get(
            ENV_WINDOWS_BACKEND_EXECUTABLE,
            DEFAULT_WINDOWS_BACKEND_EXECUTABLE,
        )
        self.center_onnx_model = os.environ.get(
            ENV_CENTER_ONNX_MODEL, DEFAULT_CENTER_ONNX_MODEL
        )
        self.ellipse_onnx_model = os.environ.get(
            ENV_ELLIPSE_ONNX_MODEL, DEFAULT_ELLIPSE_ONNX_MODEL
        )
        self.ellipse_matrix = os.environ.get(
            ENV_ELLIPSE_MATRIX, DEFAULT_ELLIPSE_MATRIX
        )
        self.selective_scan_dll = os.environ.get(
            ENV_SELECTIVE_SCAN_DLL, DEFAULT_SELECTIVE_SCAN_DLL
        )
        self.ready_timeout_s = read_float_env(
            ENV_BACKEND_READY_TIMEOUT_S,
            DEFAULT_BACKEND_READY_TIMEOUT_S,
        )
        self.log_path = self._default_backend_log_path()
        if self.runtime_kind == "windows":
            self.backend_process = WindowsBackendProcess(self.log_path)
        else:
            self.backend_process = WslBackendProcess(self.wsl_distro, self.log_path)

    @property
    def runtime_display_name(self):
        if self.runtime_kind == "windows":
            return "Windows ONNX CUDA"
        return "WSL"

    def is_running(self):
        return self.network_thread is not None and self.network_thread.isRunning()

    def start(self, weights_path, prediction_mode, port=DEFAULT_INFERENCE_PORT, host=DEFAULT_INFERENCE_HOST):
        if not weights_path:
            raise ValueError("weights_path is required")
        if prediction_mode not in ("center", "ellipse"):
            raise ValueError("prediction_mode is not set")
        self.weights_path = weights_path
        self.prediction_mode = prediction_mode

        started_backend = False
        try:
            if not self.backend_process.is_running():
                project_dir = self._runtime_root_dir()
                self.backend_process.kill_stale()
                if self.runtime_kind == "windows":
                    cmd = self._build_windows_command(
                        project_dir,
                        weights_path,
                        prediction_mode,
                        port,
                    )
                else:
                    linux_script = os.path.join(project_dir, "linux_backend.py")
                    cmd = build_backend_command(
                        self.wsl_distro,
                        self.linux_python,
                        self._to_wsl_path(linux_script),
                        self._to_wsl_path(weights_path),
                        prediction_mode,
                        port,
                    )
                self.backend_process.start(cmd, project_dir)
                started_backend = True

            wait_for_backend_ready_with_log(
                host=host,
                port=port,
                timeout_s=self.ready_timeout_s,
                backend_process=self.backend_process.process,
                log_path=self.log_path,
            )
        except Exception:
            if started_backend or self.backend_process.process is not None:
                self.stop()
            raise

        if self.network_thread is None or not self.network_thread.isRunning():
            from backend.NetworkThread import NetworkThread

            self.network_thread = NetworkThread(
                self.frame_queue,
                host=host,
                port=port,
                request_timeout_ms=DEFAULT_NETWORK_TIMEOUT_MS,
            )
            self.network_thread.result_signal.connect(self.prediction_signal.emit)
            self.network_thread.start()

    def restart(self, prediction_mode=None, port=DEFAULT_INFERENCE_PORT, host=DEFAULT_INFERENCE_HOST):
        if not self.weights_path:
            return
        mode = prediction_mode or self.prediction_mode
        weights_path = self.weights_path
        self.stop()
        self.start(weights_path, mode, port=port, host=host)

    def stop(self):
        if self.network_thread:
            self.network_thread.stop()
            if not self.network_thread.wait(2000):
                self.network_thread.terminate()
                self.network_thread.wait(500)
            self.network_thread.deleteLater()
            self.network_thread = None

        self.backend_process.stop()

    def _to_wsl_path(self, path):
        return to_wsl_path(path, self.wsl_distro)

    def _build_windows_command(
        self,
        project_dir,
        weights_path,
        prediction_mode,
        port,
    ):
        center_model_path = self._resolve_windows_path(
            self.center_onnx_model,
            project_dir,
        )
        ellipse_model_path = self._resolve_windows_path(
            self.ellipse_onnx_model,
            project_dir,
        )
        selected_path = str(weights_path)
        selected_name = os.path.basename(selected_path).lower()
        if selected_path.lower().endswith(".onnx"):
            selected_path = self._resolve_windows_path(selected_path, project_dir)
            if "ellipse" in selected_name:
                ellipse_model_path = selected_path
            elif "center" in selected_name:
                center_model_path = selected_path
            elif prediction_mode == "ellipse":
                ellipse_model_path = selected_path
            else:
                center_model_path = selected_path
        ellipse_matrix_path = self._resolve_windows_path(
            self.ellipse_matrix,
            project_dir,
        )
        custom_op_library = self._resolve_windows_path(
            self.selective_scan_dll,
            project_dir,
        )
        required = {
            "center ONNX model": center_model_path,
            "ellipse ONNX model": ellipse_model_path,
            "ellipse matrix_A": ellipse_matrix_path,
            "selective scan CUDA DLL": custom_op_library,
        }
        if self._is_frozen_ui():
            backend_executable = self._resolve_windows_path(
                self.windows_backend_executable,
                project_dir,
            )
            required = {
                "Windows backend executable": backend_executable,
                **required,
            }
        else:
            python_executable = self._resolve_windows_path(
                self.windows_python,
                project_dir,
            )
            backend_script = os.path.join(project_dir, "windows_backend.py")
            required = {
                "Windows Python": python_executable,
                "Windows backend script": backend_script,
                **required,
            }
        missing = [
            f"{label}: {path}"
            for label, path in required.items()
            if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Windows ONNX 推理文件缺失：\n" + "\n".join(missing)
            )
        self.active_model_path = (
            ellipse_model_path if prediction_mode == "ellipse" else center_model_path
        )
        if self._is_frozen_ui():
            return build_windows_backend_executable_command(
                backend_executable,
                center_model_path,
                ellipse_model_path,
                ellipse_matrix_path,
                custom_op_library,
                prediction_mode,
                port,
            )
        return build_windows_backend_command(
            python_executable,
            backend_script,
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_library,
            prediction_mode,
            port,
        )

    @staticmethod
    def _is_frozen_ui():
        return bool(getattr(sys, "frozen", False))

    @staticmethod
    def _resolve_windows_path(path, project_dir):
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        if not os.path.isabs(expanded):
            expanded = os.path.join(project_dir, expanded)
        return os.path.abspath(expanded)

    def _runtime_root_dir(self):
        return runtime_root_dir(__file__)

    def _default_backend_log_path(self):
        return default_backend_log_path(self._runtime_root_dir())
