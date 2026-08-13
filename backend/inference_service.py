import math
import os
import secrets
import sys
import threading

from backend.backend_healthcheck import (
    StartupCancelledError,
    wait_for_backend_ready_with_log,
)
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


STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_ERROR = "error"
LIFECYCLE_STATES = frozenset(
    (STATE_STOPPED, STATE_STARTING, STATE_RUNNING, STATE_STOPPING, STATE_ERROR)
)


class InferenceService:
    def __init__(
        self,
        frame_queue,
        prediction_signal=None,
        state_callback=None,
        prediction_callback=None,
    ):
        self.frame_queue = frame_queue
        self.prediction_signal = prediction_signal
        self.prediction_callback = prediction_callback
        self.network_thread = None
        self._state = STATE_STOPPED
        self._state_callback = state_callback
        self.last_error = None
        self._instance_nonce = None
        self._startup_cancelled = threading.Event()
        self._cancel_latched = False
        self._state_lock = threading.RLock()
        self._backend_operation_lock = threading.Lock()
        self._backend_operation = None
        self._backend_ready = False
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
        self.ready_timeout_s = _finite_positive_timeout(
            read_float_env(
                ENV_BACKEND_READY_TIMEOUT_S,
                DEFAULT_BACKEND_READY_TIMEOUT_S,
            ),
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

    @property
    def state(self):
        with self._state_lock:
            return self._state

    def is_running(self):
        with self._state_lock:
            network_thread = self.network_thread
        network_running = network_thread is not None and network_thread.isRunning()
        backend_running = self.backend_process.is_running()
        running = network_running and backend_running
        if self.state == STATE_RUNNING and not running:
            missing = []
            if not network_running:
                missing.append("network thread")
            if not backend_running:
                missing.append("backend process")
            self._set_state(
                STATE_ERROR,
                f"Inference runtime lost its {' and '.join(missing)}",
            )
        return running

    def start_backend(
        self,
        weights_path,
        prediction_mode,
        port=DEFAULT_INFERENCE_PORT,
        host=DEFAULT_INFERENCE_HOST,
    ):
        """Start and verify the backend process without creating Qt objects."""
        port = self._validate_start_request(weights_path, prediction_mode, port)
        with self._backend_operation_lock:
            with self._state_lock:
                if self.network_thread is not None:
                    raise RuntimeError(
                        "stop_network() must run on the UI thread before starting a backend"
                    )
            self._begin_backend_operation("start", STATE_STARTING)
            try:
                if self.backend_process.process is not None:
                    self._stop_backend_locked()
                identity = self._start_backend_locked(
                    weights_path,
                    prediction_mode,
                    port,
                    host,
                )
                self._raise_if_start_cancelled()
                with self._state_lock:
                    self._backend_ready = True
                return identity
            except Exception as exc:
                self._abort_backend_operation_locked(exc)
            finally:
                self._end_backend_operation()

    def start_network(
        self,
        host=DEFAULT_INFERENCE_HOST,
        port=DEFAULT_INFERENCE_PORT,
        start_paused=False,
    ):
        """Create/start NetworkThread; callers must invoke this on the UI thread."""
        port = self._validate_port(port)
        with self._backend_operation_lock:
            try:
                self._raise_if_start_cancelled()
            except StartupCancelledError:
                self._set_state(STATE_STOPPING)
                raise
            with self._state_lock:
                backend_ready = self._backend_ready
                thread = self.network_thread
            if thread is not None and thread.isRunning():
                if self.backend_process.is_running():
                    if start_paused:
                        thread.invalidate_generation()
                    self._commit_network_running()
                    return thread
                raise RuntimeError("Inference backend exited before network startup")
            if thread is not None:
                self._stop_network_locked()
            if not backend_ready or not self.backend_process.is_running():
                self._set_state(STATE_ERROR, "Inference backend is not ready")
                raise RuntimeError("Inference backend is not ready")

            from backend.NetworkThread import NetworkThread

            thread = NetworkThread(
                self.frame_queue,
                host=host,
                port=port,
                request_timeout_ms=DEFAULT_NETWORK_TIMEOUT_MS,
            )
            prediction_callback = getattr(self, "prediction_callback", None)
            if prediction_callback is None:
                prediction_callback = self._forward_legacy_prediction
            thread.result_signal.connect(prediction_callback)
            if start_paused:
                # The camera API installs a CONFIG payload before allowing
                # event requests through.  Pausing before QThread.start()
                # removes the small race where an old queued frame could be
                # inferred with dimensions from the previous source.
                thread.invalidate_generation()
            with self._state_lock:
                self.network_thread = thread
            try:
                thread.start()
                self._raise_if_start_cancelled()
                if not self.backend_process.is_running():
                    raise RuntimeError("Inference backend exited during network startup")
                self._commit_network_running()
            except Exception as exc:
                try:
                    self._stop_network_locked()
                except Exception as stop_exc:
                    message = f"{exc}; network cleanup failed: {stop_exc}"
                    self._set_state(STATE_ERROR, message)
                    raise RuntimeError(message) from exc
                if isinstance(exc, StartupCancelledError):
                    self._set_state(STATE_STOPPING)
                else:
                    self._set_state(STATE_ERROR, str(exc))
                raise

            return thread

    def stop_network(self):
        """Stop/destroy NetworkThread; callers must invoke this on the UI thread."""
        self._startup_cancelled.set()
        with self._backend_operation_lock:
            with self._state_lock:
                has_thread = self.network_thread is not None
            if has_thread:
                self._set_state(STATE_STOPPING)
            try:
                self._stop_network_locked()
            except Exception as exc:
                self._set_state(STATE_ERROR, str(exc))
                raise
            if not self.backend_process.is_running():
                with self._state_lock:
                    self._backend_ready = False
                self._clear_start_cancellation()
                self._set_state(STATE_STOPPED)
            return True

    def _forward_legacy_prediction(self, result, timestamp, _generation):
        """Compatibility adapter for non-UI callers using the old signal API."""
        if self.prediction_signal is not None:
            self.prediction_signal.emit(result, timestamp)

    def stop_backend(self):
        """Stop the backend process without touching NetworkThread."""
        self._startup_cancelled.set()
        with self._backend_operation_lock:
            with self._state_lock:
                if self.network_thread is not None:
                    raise RuntimeError(
                        "stop_network() must run on the UI thread before stopping a backend"
                    )
            self._begin_backend_operation("stop", STATE_STOPPING, clear_cancel=False)
            try:
                self._stop_backend_locked()
            except Exception as exc:
                self._set_state(STATE_ERROR, str(exc))
                raise
            finally:
                self._end_backend_operation()

            self._clear_start_cancellation()
            self._set_state(STATE_STOPPED)
            return True

    def restart_backend(
        self,
        prediction_mode=None,
        port=DEFAULT_INFERENCE_PORT,
        host=DEFAULT_INFERENCE_HOST,
    ):
        """Restart only the backend; NetworkThread must already be stopped."""
        if not self.weights_path:
            return None
        mode = prediction_mode or self.prediction_mode
        port = self._validate_start_request(self.weights_path, mode, port)
        with self._backend_operation_lock:
            with self._state_lock:
                if self.network_thread is not None:
                    raise RuntimeError(
                        "stop_network() must run on the UI thread before restarting a backend"
                    )
            self._begin_backend_operation("restart", STATE_STOPPING)
            try:
                self._stop_backend_locked()
                self._raise_if_start_cancelled()
                self._set_state(STATE_STARTING)
                identity = self._start_backend_locked(
                    self.weights_path,
                    mode,
                    port,
                    host,
                )
                self._raise_if_start_cancelled()
                with self._state_lock:
                    self._backend_ready = True
                return identity
            except Exception as exc:
                self._abort_backend_operation_locked(exc)
            finally:
                self._end_backend_operation()

    def start(self, weights_path, prediction_mode, port=DEFAULT_INFERENCE_PORT, host=DEFAULT_INFERENCE_HOST):
        """Compatibility wrapper for callers that run all phases on one thread."""
        if not weights_path:
            raise ValueError("weights_path is required")
        if self.is_running():
            return
        with self._state_lock:
            has_network = self.network_thread is not None
        if has_network or self.backend_process.process is not None:
            self.stop()
        try:
            self.start_backend(weights_path, prediction_mode, port=port, host=host)
            self.start_network(host=host, port=port)
        except Exception as exc:
            try:
                self.stop()
            except Exception as stop_exc:
                message = f"{exc}; cleanup failed: {stop_exc}"
                self._set_state(STATE_ERROR, message)
                raise RuntimeError(message) from exc
            raise

    def cancel_start(self, force=False):
        with self._state_lock:
            cancellable = (
                self._state == STATE_STARTING
                or self._backend_operation in ("start", "restart")
                or bool(force)
            )
            if not cancellable:
                return False
            # Latch an explicit UI cancellation separately from the event
            # used internally by stop_network(). A backend worker may not yet
            # have entered _begin_backend_operation; without this latch its
            # normal clear step could erase a close request in that tiny gap.
            self._cancel_latched = True
            self._startup_cancelled.set()
            return True

    def restart(self, prediction_mode=None, port=DEFAULT_INFERENCE_PORT, host=DEFAULT_INFERENCE_HOST):
        """Compatibility wrapper for callers that run all phases on one thread."""
        if not self.weights_path:
            return
        self.stop_network()
        try:
            self.restart_backend(prediction_mode, port=port, host=host)
            self.start_network(host=host, port=port)
        except Exception as exc:
            try:
                self.stop()
            except Exception as stop_exc:
                message = f"{exc}; cleanup failed: {stop_exc}"
                self._set_state(STATE_ERROR, message)
                raise RuntimeError(message) from exc
            raise

    def stop(self):
        """Compatibility wrapper for callers that run all phases on one thread."""
        self._startup_cancelled.set()
        errors = []
        try:
            self.stop_network()
        except Exception as exc:
            errors.append(exc)
        try:
            self.stop_backend()
        except Exception as exc:
            errors.append(exc)
        if errors:
            message = "; ".join(str(error) for error in errors)
            self._set_state(STATE_ERROR, message)
            raise RuntimeError(f"Failed to stop inference service: {message}")
        return True

    def _validate_start_request(self, weights_path, prediction_mode, port):
        if not weights_path:
            raise ValueError("weights_path is required")
        if prediction_mode not in ("center", "ellipse"):
            raise ValueError("prediction_mode is not set")
        return self._validate_port(port)

    @staticmethod
    def _validate_port(port):
        try:
            port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError("port must be an integer between 1 and 65535") from exc
        if not 1 <= port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        return port

    def _begin_backend_operation(self, operation, state, clear_cancel=True):
        with self._state_lock:
            if self._backend_operation is not None:
                raise RuntimeError(
                    f"Inference backend operation already active: {self._backend_operation}"
                )
            if clear_cancel and not self._cancel_latched:
                self._startup_cancelled.clear()
            self._backend_operation = operation
        self._set_state(state)

    def _end_backend_operation(self):
        with self._state_lock:
            self._backend_operation = None

    def _raise_if_start_cancelled(self):
        if self._startup_cancelled.is_set():
            raise StartupCancelledError()

    def _clear_start_cancellation(self):
        with self._state_lock:
            self._startup_cancelled.clear()
            self._cancel_latched = False

    def _start_backend_locked(self, weights_path, prediction_mode, port, host):
        self.weights_path = weights_path
        self.prediction_mode = prediction_mode
        project_dir = self._runtime_root_dir()
        self.backend_process.kill_stale()
        self._raise_if_start_cancelled()

        instance_nonce = secrets.token_hex(16)
        self._instance_nonce = instance_nonce
        if self.runtime_kind == "windows":
            cmd = self._build_windows_command(
                project_dir,
                weights_path,
                prediction_mode,
                port,
                instance_nonce,
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
                instance_nonce,
            )

        process = self.backend_process.start(
            cmd,
            project_dir,
            instance_nonce=instance_nonce,
        )
        expected_pid = None
        if self.runtime_kind == "windows":
            expected_pid = getattr(process, "pid", None)
        try:
            identity = wait_for_backend_ready_with_log(
                host=host,
                port=port,
                timeout_s=self.ready_timeout_s,
                backend_process=self.backend_process.process,
                log_path=self.log_path,
                expected_nonce=instance_nonce,
                expected_pid=expected_pid,
                cancel_event=self._startup_cancelled,
            )
        except StartupCancelledError as exc:
            # A WSL READY reply carries the remote Python PID needed for an
            # exact cleanup.  Preserve it even when cancellation wins.
            if exc.identity is not None:
                self.backend_process.record_backend_identity(exc.identity)
            raise

        self.backend_process.record_backend_identity(identity)
        self._raise_if_start_cancelled()
        if not self.backend_process.is_running():
            raise RuntimeError("Inference backend exited after reporting READY")
        return identity

    def _stop_backend_locked(self):
        with self._state_lock:
            self._backend_ready = False
        self.backend_process.stop()
        self._instance_nonce = None

    def _stop_network_locked(self):
        with self._state_lock:
            thread = self.network_thread
        if thread is None:
            return True

        thread.stop()
        if thread.isRunning() and not thread.wait(2000):
            thread.terminate()
            thread.wait(500)
        if thread.isRunning():
            # Keep the live handle reachable so a later retry can stop it.
            raise RuntimeError("Inference network thread did not stop")
        thread.deleteLater()
        with self._state_lock:
            if self.network_thread is thread:
                self.network_thread = None
        return True

    def _abort_backend_operation_locked(self, error):
        cleanup_error = None
        try:
            self._stop_backend_locked()
        except Exception as exc:
            cleanup_error = exc

        if cleanup_error is not None:
            message = f"{error}; backend cleanup failed: {cleanup_error}"
            self._set_state(STATE_ERROR, message)
            raise RuntimeError(message) from error
        if isinstance(error, StartupCancelledError):
            self._clear_start_cancellation()
            self._set_state(STATE_STOPPED)
        else:
            self._set_state(STATE_ERROR, str(error))
        raise error

    def _commit_network_running(self):
        callback = None
        with self._state_lock:
            if self._startup_cancelled.is_set():
                raise StartupCancelledError()
            self._startup_cancelled.clear()
            self._cancel_latched = False
            self._state = STATE_RUNNING
            self.last_error = None
            callback = self._state_callback
        if callback is not None:
            try:
                callback(STATE_RUNNING, None)
            except Exception:
                pass

    def _to_wsl_path(self, path):
        return to_wsl_path(path, self.wsl_distro)

    def _build_windows_command(
        self,
        project_dir,
        weights_path,
        prediction_mode,
        port,
        instance_nonce=None,
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
                instance_nonce,
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
            instance_nonce,
        )

    def _set_state(self, state, error=None):
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"Unknown inference lifecycle state: {state}")
        with self._state_lock:
            self._state = state
            self.last_error = str(error) if error else None
            callback = self._state_callback
            last_error = self.last_error
        if callback is not None:
            try:
                callback(state, last_error)
            except Exception:
                # Lifecycle reporting must not break process cleanup.
                pass

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


def _finite_positive_timeout(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value <= 0:
        return float(default)
    return value
