import os

from backend.backend_healthcheck import wait_for_backend_ready_with_log
from backend.inference_runtime import default_backend_log_path, read_float_env, runtime_root_dir, to_wsl_path
from backend.NetworkThread import NetworkThread
from backend.settings import (
    DEFAULT_BACKEND_READY_TIMEOUT_S,
    DEFAULT_INFERENCE_HOST,
    DEFAULT_INFERENCE_PORT,
    DEFAULT_LINUX_PYTHON,
    DEFAULT_NETWORK_TIMEOUT_MS,
    DEFAULT_WSL_DISTRO,
    ENV_BACKEND_READY_TIMEOUT_S,
    ENV_LINUX_PYTHON,
    ENV_WSL_DISTRO,
)
from backend.wsl_process import WslBackendProcess, build_backend_command


class InferenceService:
    def __init__(self, frame_queue, prediction_signal):
        self.frame_queue = frame_queue
        self.prediction_signal = prediction_signal
        self.network_thread = None
        self.weights_path = None
        self.prediction_mode = "center"
        self.wsl_distro = os.environ.get(ENV_WSL_DISTRO, DEFAULT_WSL_DISTRO)
        self.linux_python = os.environ.get(ENV_LINUX_PYTHON, DEFAULT_LINUX_PYTHON)
        self.ready_timeout_s = read_float_env(ENV_BACKEND_READY_TIMEOUT_S, DEFAULT_BACKEND_READY_TIMEOUT_S)
        self.log_path = self._default_backend_log_path()
        self.backend_process = WslBackendProcess(self.wsl_distro, self.log_path)

    def is_running(self):
        return self.network_thread is not None and self.network_thread.isRunning()

    def start(self, weights_path, prediction_mode, port=DEFAULT_INFERENCE_PORT, host=DEFAULT_INFERENCE_HOST):
        if not weights_path:
            raise ValueError("weights_path is required")
        if prediction_mode not in ("center", "ellipse"):
            raise ValueError("prediction_mode is not set")

        self.weights_path = weights_path
        self.prediction_mode = prediction_mode
        wsl_weights_path = self._to_wsl_path(weights_path)

        started_backend = False
        try:
            if not self.backend_process.is_running():
                project_dir = self._runtime_root_dir()
                linux_script = os.path.join(project_dir, "linux_backend.py")
                wsl_linux_script = self._to_wsl_path(linux_script)
                self.backend_process.kill_stale()
                cmd = build_backend_command(
                    self.wsl_distro,
                    self.linux_python,
                    wsl_linux_script,
                    wsl_weights_path,
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

    def _runtime_root_dir(self):
        return runtime_root_dir(__file__)

    def _default_backend_log_path(self):
        return default_backend_log_path(self._runtime_root_dir())
