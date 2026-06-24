import os
import subprocess
import time


def build_backend_command(wsl_distro, linux_python, linux_script, weights_path, prediction_mode, port):
    weights_arg = "--ellipse-weights" if prediction_mode == "ellipse" else "--center-weights"
    return [
        "wsl",
        "-d",
        wsl_distro,
        linux_python,
        linux_script,
        weights_arg,
        weights_path,
        "--port",
        str(port),
    ]


class WslBackendProcess:
    def __init__(self, wsl_distro, log_path):
        self.wsl_distro = wsl_distro
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
        self.process._codex_log_handle = backend_log
        return self.process

    def stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self._close_log_handle()
            self.process = None

        kill_stale_backend_processes(self.wsl_distro)

    def kill_stale(self):
        kill_stale_backend_processes(self.wsl_distro)

    def _close_log_handle(self):
        log_handle = getattr(self.process, "_codex_log_handle", None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass


def kill_stale_backend_processes(wsl_distro, delay_s=0.2):
    try:
        subprocess.run(
            ["wsl", "-d", wsl_distro, "pkill", "-f", "linux_backend.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        if delay_s:
            time.sleep(delay_s)
    except Exception:
        pass
