import secrets
import sys
import threading

from backend.backend_healthcheck import (
    StartupCancelledError,
    wait_for_backend_ready_with_log,
)
from backend.inference_backend_runtime import (
    InferenceRuntimeSettings,
    build_windows_launch,
    build_wsl_launch,
    finite_positive_timeout,
    resolve_runtime_path,
)
from backend.inference_lifecycle import (
    LIFECYCLE_STATES,
    STATE_ERROR,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    validate_lifecycle_state,
)
from backend.inference_runtime import default_backend_log_path, runtime_root_dir, to_wsl_path
from backend.inference_transport import create_network_thread, stop_network_thread
from backend.model_contract import is_supported_model_mode
from backend.settings import (
    DEFAULT_INFERENCE_HOST,
    DEFAULT_INFERENCE_PORT,
    DEFAULT_NETWORK_TIMEOUT_MS,
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
        runtime = InferenceRuntimeSettings.from_environment()
        self.runtime_settings = runtime
        self.runtime_kind = runtime.kind
        self.wsl_distro = runtime.wsl_distro
        self.linux_python = runtime.linux_python
        self.windows_python = runtime.windows_python
        self.windows_backend_executable = runtime.windows_backend_executable
        self.center_onnx_model = runtime.center_onnx_model
        self.ellipse_onnx_model = runtime.ellipse_onnx_model
        self.ellipse_matrix = runtime.ellipse_matrix
        self.selective_scan_dll = runtime.selective_scan_dll
        self.ready_timeout_s = runtime.ready_timeout_s
        self.log_path = self._default_backend_log_path()
        self.backend_process = runtime.create_process(self.log_path)

    @property
    def runtime_display_name(self):
        runtime = getattr(self, "runtime_settings", None)
        if runtime is not None and runtime.kind == self.runtime_kind:
            return runtime.display_name
        return "Windows ONNX CUDA" if self.runtime_kind == "windows" else "WSL"

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

            prediction_callback = getattr(self, "prediction_callback", None)
            if prediction_callback is None:
                prediction_callback = self._forward_legacy_prediction
            thread = create_network_thread(
                self.frame_queue,
                host,
                port,
                DEFAULT_NETWORK_TIMEOUT_MS,
                prediction_callback,
                start_paused=start_paused,
            )
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
        if not is_supported_model_mode(prediction_mode):
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
            launch = build_wsl_launch(
                project_dir,
                weights_path,
                prediction_mode,
                port,
                instance_nonce,
                distro=self.wsl_distro,
                linux_python=self.linux_python,
                path_converter=self._to_wsl_path,
            )
            cmd = launch.command
            self.active_model_path = launch.active_model_path

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

        # The helper raises before deleteLater when termination fails, so the
        # live handle remains reachable for a later stop retry.
        stop_network_thread(thread)
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
        launch = build_windows_launch(
            project_dir,
            weights_path,
            prediction_mode,
            port,
            instance_nonce,
            center_model=self.center_onnx_model,
            ellipse_model=self.ellipse_onnx_model,
            ellipse_matrix=self.ellipse_matrix,
            custom_op_library=self.selective_scan_dll,
            python_executable=self.windows_python,
            backend_executable=self.windows_backend_executable,
            frozen=self._is_frozen_ui(),
        )
        self.active_model_path = launch.active_model_path
        return launch.command

    def _set_state(self, state, error=None):
        validate_lifecycle_state(state)
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
        return resolve_runtime_path(path, project_dir)

    def _runtime_root_dir(self):
        return runtime_root_dir(__file__)

    def _default_backend_log_path(self):
        return default_backend_log_path(self._runtime_root_dir())


def _finite_positive_timeout(value, default):
    """Compatibility export; new code uses the runtime settings helper."""
    return finite_positive_timeout(value, default)
