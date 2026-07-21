#Windows端
import pickle
import pickle
import queue
import zmq
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkThread(QThread):
    result_signal = pyqtSignal(object, int)

    def __init__(self, input_queue, host="127.0.0.1", port=5555, request_timeout_ms=1000):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self.endpoint = f"tcp://{host}:{port}"
        self.request_timeout_ms = request_timeout_ms
        self.context = None
        self.socket = None
        self._last_error = None

    def run(self):
        """与本机或 WSL 推理服务通信。"""
        self.context = zmq.Context()
        self._open_socket()
        try:
            while self.running:
                data = self._get_latest_payload()
                if data is None:
                    continue

                timestamp = 0
                if isinstance(data, dict) and "timestamp" in data:
                    timestamp = int(data["timestamp"])

                try:
                    self.socket.send_pyobj(data)
                    result = self._recv_result()
                    self._last_error = None
                    self.result_signal.emit(result, timestamp)
                except zmq.Again:
                    self._emit_error_once("通信超时：请确认推理服务已启动", timestamp)
                    self._reset_socket()
                except zmq.ZMQError as exc:
                    if self.running:
                        self._emit_error_once(f"通信异常：{exc}", timestamp)
                        self._reset_socket()
                except Exception as exc:
                    if self.running:
                        self._emit_error_once(f"响应解析异常：{exc}", timestamp)
                        self._reset_socket()
        finally:
            self._close_socket()
            if self.context is not None:
                self.context.term()
                self.context = None

    def _get_latest_payload(self):
        try:
            data = self.input_queue.get(timeout=0.2)
        except queue.Empty:
            return None

        while True:
            try:
                data = self.input_queue.get_nowait()
            except queue.Empty:
                return data

    def _open_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        try:
            self.socket.setsockopt(zmq.IMMEDIATE, 1)
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        except (AttributeError, zmq.ZMQError):
            pass
        self.socket.connect(self.endpoint)

    def _recv_result(self):
        raw = self.socket.recv()
        try:
            return pickle.loads(raw)
        except Exception:
            return raw.decode("utf-8", errors="replace")

    def _close_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def _reset_socket(self):
        self._close_socket()
        if self.running:
            self._open_socket()

    def _emit_error_once(self, message, timestamp):
        if message != self._last_error:
            self.result_signal.emit(message, timestamp)
            self._last_error = message

    def stop(self):
        self.running = False
