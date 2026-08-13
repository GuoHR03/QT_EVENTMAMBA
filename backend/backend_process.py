import os
import subprocess


class BackendProcess:
    """Common subprocess lifecycle for inference backends."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.process = None
        self._log_handle = None
        self.backend_pid = None
        self.instance_nonce = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, cmd, cwd, instance_nonce=None):
        if self.is_running():
            raise RuntimeError("Inference backend process is already running")
        if self.process is not None:
            # Reap an already-exited child before replacing its handle.
            self.process.wait(timeout=0)
            self.process = None
            self._close_log_handle()

        log_directory = os.path.dirname(self.log_path)
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)
        self._log_handle = open(self.log_path, "w", encoding="utf-8")
        self.backend_pid = None
        self.instance_nonce = instance_nonce
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=child_env,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except Exception:
            self._close_log_handle()
            self._clear_identity()
            raise
        return self.process

    def stop(self):
        process = self.process
        if process is None:
            self._close_log_handle()
            self._clear_identity()
            return True

        if process.poll() is not None:
            try:
                process.wait(timeout=0)
            finally:
                self._close_log_handle()
                self.process = None
                self._clear_identity()
            return True

        errors = []
        exited = False
        try:
            process.terminate()
            process.wait(timeout=2)
            exited = True
        except Exception as exc:
            errors.append(exc)
            try:
                process.kill()
                process.wait(timeout=1)
                exited = True
            except Exception as kill_exc:
                errors.append(kill_exc)

        if not exited and process.poll() is not None:
            exited = True

        if exited:
            self._close_log_handle()
            self.process = None
            self._clear_identity()
            return True

        # Keep both the live process and its log handle reachable so a later
        # retry can terminate it. Silently dropping these handles would leak a
        # backend that can continue owning the inference port.
        details = "; ".join(str(error) for error in errors if str(error))
        suffix = f": {details}" if details else ""
        raise RuntimeError(f"Inference backend process did not stop{suffix}")

    def kill_stale(self):
        return False

    def record_backend_identity(self, identity):
        if not isinstance(identity, dict):
            return
        pid = identity.get("pid")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return
        if pid > 0:
            self.backend_pid = pid

    def _clear_identity(self):
        self.backend_pid = None
        self.instance_nonce = None

    def _close_log_handle(self):
        if self._log_handle is None:
            return
        try:
            self._log_handle.close()
        finally:
            self._log_handle = None
