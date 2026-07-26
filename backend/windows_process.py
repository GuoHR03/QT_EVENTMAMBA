import os
import subprocess


def _windows_backend_arguments(
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
):
    return [
        "--center-model",
        center_model_path,
        "--ellipse-model",
        ellipse_model_path,
        "--ellipse-matrix",
        ellipse_matrix_path,
        "--custom-op-library",
        custom_op_library,
        "--initial-mode",
        initial_mode,
        "--port",
        str(port),
    ]


def build_windows_backend_command(
    python_executable,
    backend_script,
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
):
    return [
        python_executable,
        backend_script,
        *_windows_backend_arguments(
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_library,
            initial_mode,
            port,
        ),
    ]


def build_windows_backend_executable_command(
    backend_executable,
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
):
    return [
        backend_executable,
        *_windows_backend_arguments(
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_library,
            initial_mode,
            port,
        ),
    ]


class WindowsBackendProcess:
    def __init__(self, log_path):
        self.log_path = log_path
        self.process = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, cmd, cwd):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        backend_log = open(self.log_path, "w", encoding="utf-8")
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.process = subprocess.Popen(
            cmd,
            stdout=backend_log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=child_env,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        self.process._eventmamba_log_handle = backend_log
        return self.process

    def stop(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            try:
                self.process.kill()
                self.process.wait(timeout=1)
            except Exception:
                pass
        self._close_log_handle()
        self.process = None

    def kill_stale(self):
        # Do not kill unrelated local Python processes. A stale server will make
        # the new backend fail its port bind and the health check will show its log.
        return None

    def _close_log_handle(self):
        log_handle = getattr(self.process, "_eventmamba_log_handle", None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass
