import math
import subprocess
import time

from backend.backend_process import BackendProcess


def build_backend_command(
    wsl_distro,
    linux_python,
    linux_script,
    weights_path,
    prediction_mode,
    port,
    instance_nonce=None,
):
    weights_arg = (
        "--ellipse-weights" if prediction_mode == "ellipse" else "--center-weights"
    )
    command = [
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
    if instance_nonce is not None:
        command.extend(("--instance-nonce", str(instance_nonce)))
    return command


class WslBackendProcess(BackendProcess):
    def __init__(self, wsl_distro, log_path):
        super().__init__(log_path)
        self.wsl_distro = wsl_distro
        self.remote_pid = None
        self.remote_nonce = None

    def start(self, cmd, cwd, instance_nonce=None):
        self.remote_pid = None
        self.remote_nonce = None
        return super().start(cmd, cwd, instance_nonce=instance_nonce)

    def record_backend_identity(self, identity):
        super().record_backend_identity(identity)
        if not isinstance(identity, dict):
            return
        try:
            pid = int(identity.get("pid"))
        except (TypeError, ValueError):
            return
        nonce = identity.get("instance_nonce")
        if pid > 0 and nonce and nonce == self.instance_nonce:
            self.remote_pid = pid
            self.remote_nonce = nonce

    def stop(self):
        remote_error = None
        if self.remote_pid is not None:
            try:
                kill_stale_backend_processes(
                    self.wsl_distro,
                    self.remote_pid,
                    self.remote_nonce,
                )
            except Exception as exc:
                remote_error = exc

        local_error = None
        try:
            super().stop()
        except Exception as exc:
            local_error = exc

        if remote_error is None and local_error is None:
            self.remote_pid = None
            self.remote_nonce = None
            return True

        errors = [error for error in (remote_error, local_error) if error is not None]
        details = "; ".join(str(error) for error in errors)
        raise RuntimeError(f"WSL inference backend did not stop: {details}")

    def kill_stale(self):
        if self.remote_pid is None:
            return False
        kill_stale_backend_processes(
            self.wsl_distro,
            self.remote_pid,
            self.remote_nonce,
        )
        self.remote_pid = None
        self.remote_nonce = None
        return True


def kill_stale_backend_processes(
    wsl_distro,
    pid=None,
    instance_nonce=None,
    timeout_s=2.0,
    poll_interval_s=0.05,
    delay_s=None,
):
    """Stop one verified WSL backend; never scan or pkill unrelated jobs."""
    if pid is None or not instance_nonce:
        return False
    try:
        pid = int(pid)
        timeout_s = float(timeout_s)
        poll_interval_s = float(poll_interval_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("WSL PID and timeouts must be numeric") from exc
    if (
        pid <= 0
        or not math.isfinite(timeout_s)
        or not math.isfinite(poll_interval_s)
        or timeout_s <= 0
        or poll_interval_s <= 0
    ):
        raise ValueError("WSL PID and timeouts must be positive")

    cmdline = _run_wsl(
        wsl_distro,
        ["cat", f"/proc/{pid}/cmdline"],
        timeout_s=min(timeout_s, 2.0),
        capture_output=True,
    )
    if cmdline.returncode != 0:
        return False
    command_parts = cmdline.stdout.split(b"\0")
    if str(instance_nonce).encode("utf-8") not in command_parts:
        raise RuntimeError(
            f"Refusing to stop WSL PID {pid}: instance nonce is not in its command line"
        )

    _run_wsl(
        wsl_distro,
        ["kill", "-TERM", str(pid)],
        timeout_s=min(timeout_s, 2.0),
    )
    if _wait_for_wsl_pid_exit(
        wsl_distro,
        pid,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    ):
        return True

    _run_wsl(
        wsl_distro,
        ["kill", "-KILL", str(pid)],
        timeout_s=min(timeout_s, 2.0),
    )
    if _wait_for_wsl_pid_exit(
        wsl_distro,
        pid,
        timeout_s=min(timeout_s, 1.0),
        poll_interval_s=poll_interval_s,
    ):
        return True
    raise RuntimeError(f"Tracked WSL backend PID {pid} is still running")


def _wait_for_wsl_pid_exit(wsl_distro, pid, timeout_s, poll_interval_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        probe = _run_wsl(
            wsl_distro,
            ["kill", "-0", str(pid)],
            # Starting wsl.exe commonly takes longer than a 50 ms polling
            # interval. Keep the process probe timeout separate from the
            # interval between successful probes.
            timeout_s=max(0.1, min(1.0, remaining)),
        )
        if probe.returncode != 0:
            return True
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval_s, remaining))
    return False


def _run_wsl(wsl_distro, command, timeout_s, capture_output=False):
    kwargs = {
        "stdout": subprocess.PIPE if capture_output else subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": timeout_s,
        "check": False,
    }
    try:
        return subprocess.run(
            ["wsl", "-d", wsl_distro, "--", *command],
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to control tracked WSL inference backend: {exc}"
        ) from exc
