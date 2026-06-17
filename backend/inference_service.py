import os
import subprocess
import sys
import time

import zmq

from backend.NetworkThread import NetworkThread


class InferenceService:
    def __init__(self, frame_queue, prediction_signal):
        self.frame_queue = frame_queue
        self.prediction_signal = prediction_signal
        self.network_thread = None
        self.backend_process = None
        self.weights_path = None
        self.prediction_mode = "center"
        self.wsl_distro = os.environ.get("EVENTMAMBA_WSL_DISTRO", "EventMamba_mini")
        self.linux_python = os.environ.get(
            "EVENTMAMBA_LINUX_PYTHON",
            "/opt/miniconda3/envs/eventmamba/bin/python",
        )
        self.ready_timeout_s = self._read_float_env("EVENTMAMBA_BACKEND_READY_TIMEOUT_S", 180.0)
        self.log_path = self._default_backend_log_path()

    def is_running(self):
        return self.network_thread is not None and self.network_thread.isRunning()

    def start(self, weights_path, prediction_mode, port=5555, host="127.0.0.1"):
        if not weights_path:
            raise ValueError("weights_path is required")
        if prediction_mode not in ("center", "ellipse"):
            raise ValueError("prediction_mode is not set")

        self.weights_path = weights_path
        self.prediction_mode = prediction_mode
        wsl_weights_path = self._to_wsl_path(weights_path)

        started_backend = False
        try:
            if self.backend_process is None or self.backend_process.poll() is not None:
                project_dir = self._runtime_root_dir()
                linux_script = os.path.join(project_dir, "linux_backend.py")
                wsl_linux_script = self._to_wsl_path(linux_script)
                weights_arg = "--ellipse-weights" if prediction_mode == "ellipse" else "--center-weights"
                self._kill_stale_backend_processes()
                cmd = [
                    "wsl",
                    "-d",
                    self.wsl_distro,
                    self.linux_python,
                    wsl_linux_script,
                    weights_arg,
                    wsl_weights_path,
                    "--port",
                    str(port),
                ]
                self.backend_process = self._start_backend_process(cmd, project_dir)
                started_backend = True

            self._wait_for_backend_ready_with_log(
                host=host,
                port=port,
                timeout_s=self.ready_timeout_s,
            )
        except Exception:
            if started_backend or self.backend_process is not None:
                self.stop()
            raise

        if self.network_thread is None or not self.network_thread.isRunning():
            self.network_thread = NetworkThread(self.frame_queue, host=host, port=port, request_timeout_ms=1000)
            self.network_thread.result_signal.connect(self.prediction_signal.emit)
            self.network_thread.start()

    def restart(self, prediction_mode=None, port=5555, host="127.0.0.1"):
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

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=2)
            except Exception:
                try:
                    self.backend_process.kill()
                except Exception:
                    pass
            log_handle = getattr(self.backend_process, "_codex_log_handle", None)
            if log_handle:
                try:
                    log_handle.close()
                except Exception:
                    pass
            self.backend_process = None

        try:
            subprocess.run(
                ["wsl", "-d", self.wsl_distro, "pkill", "-f", "linux_backend.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            pass

    def _start_backend_process(self, cmd, project_dir):
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
        process = subprocess.Popen(
            cmd,
            stdout=backend_log,
            stderr=subprocess.STDOUT,
            cwd=project_dir,
            env=child_env,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        process._codex_log_handle = backend_log
        return process

    def _kill_stale_backend_processes(self):
        try:
            subprocess.run(
                ["wsl", "-d", self.wsl_distro, "pkill", "-f", "linux_backend.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            time.sleep(0.2)
        except Exception:
            pass

    def _to_wsl_path(self, path):
        if not path:
            return path
        if path.startswith("\\\\wsl$\\"):
            parts = path.split("\\")
            if len(parts) >= 4:
                distro = parts[2]
                inner = "/".join(parts[3:])
                if distro == self.wsl_distro:
                    return f"/{inner}".replace("\\", "/")
                return f"/mnt/wsl/{distro}/{inner}".replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            rest = path[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
        return path

    def _runtime_root_dir(self):
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return getattr(sys, "_MEIPASS")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)

    def _default_backend_log_path(self):
        preferred_root = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or self._runtime_root_dir()
        log_dir = os.path.join(preferred_root, "UI_Event")
        return os.path.join(log_dir, "eventmamba_backend.log")

    def _wait_for_backend_ready_with_log(self, host, port, timeout_s=10.0, poll_interval_s=0.2):
        endpoint = f"tcp://{host}:{port}"
        deadline = time.time() + timeout_s
        context = zmq.Context()
        try:
            while time.time() < deadline:
                if self.backend_process is not None and self.backend_process.poll() is not None:
                    details = self._read_backend_log_tail()
                    if details:
                        raise RuntimeError(f"WSL 推理服务启动失败，进程已退出。\n后端日志：\n{details}")
                    raise RuntimeError("WSL 推理服务启动失败，进程已退出")

                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, int(poll_interval_s * 1000))
                socket.setsockopt(zmq.SNDTIMEO, int(poll_interval_s * 1000))
                try:
                    socket.connect(endpoint)
                    socket.send_pyobj({"msg_type": "PING"})
                    reply = socket.recv_string()
                    if reply == "READY":
                        return
                except zmq.Again:
                    time.sleep(poll_interval_s)
                except zmq.ZMQError:
                    time.sleep(poll_interval_s)
                finally:
                    socket.close(linger=0)

            details = self._read_backend_log_tail()
            if details:
                raise TimeoutError(f"等待 WSL 推理服务就绪超时。\n后端日志：\n{details}")
            raise TimeoutError("等待 WSL 推理服务就绪超时")
        finally:
            context.term()

    def _read_backend_log_tail(self, max_chars=4000):
        try:
            if not os.path.exists(self.log_path):
                return ""
            with open(self.log_path, "rb") as handle:
                raw = handle.read()
            content = self._decode_backend_log(raw)
            return content[-max_chars:].strip()
        except Exception:
            return ""

    @staticmethod
    def _decode_backend_log(raw):
        for encoding in ("utf-8", "utf-16", "utf-16-le", "gbk"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _read_float_env(name, default):
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
