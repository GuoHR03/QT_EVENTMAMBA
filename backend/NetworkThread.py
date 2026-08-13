# Windows inference client
import queue
from threading import Event, Lock

import zmq
from PyQt6.QtCore import QThread, pyqtSignal

from backend.protocol import LOCAL_ROI_CONTEXT
from backend.zmq_protocol import (
    configure_socket_limits,
    receive_response,
    send_request,
)


class NetworkThread(QThread):
    # Keep the opaque generation token on the local Qt hop.  The UI boundary
    # performs a second check, so a reply that was already queued when a
    # camera/ROI transition starts cannot repopulate stale predictions.
    result_signal = pyqtSignal(object, int, object)

    def __init__(self, input_queue, host="127.0.0.1", port=5555, request_timeout_ms=1000):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self.endpoint = f"tcp://{host}:{port}"
        self.request_timeout_ms = request_timeout_ms
        self.context = None
        self.socket = None
        self._last_error = None
        self._generation = object()
        self._generation_lock = Lock()
        self._generation_enabled = Event()
        self._generation_enabled.set()
        self._priority_payload = None

    def run(self):
        """与本机或 WSL 推理服务通信。"""
        self.context = zmq.Context()
        self._open_socket()
        try:
            while self.running:
                if not self._generation_enabled.wait(timeout=0.2):
                    continue
                data, request_generation = self._get_latest_payload_for_generation()
                if data is None:
                    continue

                timestamp = 0
                if isinstance(data, dict) and "timestamp" in data:
                    timestamp = int(data["timestamp"])

                try:
                    request_context = self._send_payload(data)
                    result = self._recv_result()
                    result = self._attach_request_context(result, request_context)
                    self._emit_result_if_current(
                        result,
                        timestamp,
                        request_generation,
                    )
                except zmq.Again:
                    self._emit_error_once(
                        "通信超时：请确认推理服务已启动",
                        timestamp,
                        request_generation,
                    )
                    self._reset_socket()
                except zmq.ZMQError as exc:
                    if self.running:
                        self._emit_error_once(
                            f"通信异常：{exc}",
                            timestamp,
                            request_generation,
                        )
                        self._reset_socket()
                except Exception as exc:
                    if self.running:
                        self._emit_error_once(
                            f"响应解析异常：{exc}",
                            timestamp,
                            request_generation,
                        )
                        self._reset_socket()
        finally:
            self._close_socket()
            if self.context is not None:
                self.context.term()
                self.context = None

    def _get_latest_payload(self):
        try:
            return self.input_queue.get(timeout=0.2)
        except queue.Empty:
            return None

    def _get_latest_payload_for_generation(self):
        # Queue.get() may wait for 200 ms.  It must stay outside the lock used
        # by the UI thread's synchronous invalidate_generation() call.
        with self._generation_lock:
            if not self.running or not self._generation_enabled.is_set():
                return None, None
            if self._priority_payload is not None:
                data = self._priority_payload
                self._priority_payload = None
                return data, self._generation
            generation = self._generation

        data = self._get_latest_payload()
        if data is None:
            return None, None

        with self._generation_lock:
            if (
                not self.running
                or not self._generation_enabled.is_set()
                or generation is not self._generation
            ):
                return None, None
            if self._priority_payload is not None:
                data = self._priority_payload
                self._priority_payload = None
                generation = self._generation
            return data, generation

    def invalidate_generation(self):
        """Discard in-flight replies and pause dequeuing until the new source is ready."""
        with self._generation_lock:
            self._generation = object()
            self._generation_enabled.clear()
            self._priority_payload = None
            self._last_error = None
            self._discard_pending_input()
            return self._generation

    def replace_pending_payload(self, payload):
        """Install a control payload that is consumed before queued event data."""
        with self._generation_lock:
            self._priority_payload = payload

    def resume_generation(self):
        with self._generation_lock:
            self._generation_enabled.set()

    def _is_current_generation(self, generation):
        with self._generation_lock:
            return generation is self._generation

    def is_generation_current(self, generation):
        return self._is_current_generation(generation)

    def _discard_pending_input(self):
        while True:
            try:
                self.input_queue.get_nowait()
            except (AttributeError, queue.Empty):
                return

    def _open_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        configure_socket_limits(self.socket)
        try:
            self.socket.setsockopt(zmq.IMMEDIATE, 1)
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        except (AttributeError, zmq.ZMQError):
            pass
        self.socket.connect(self.endpoint)

    def _recv_result(self):
        return receive_response(self.socket)

    def _send_payload(self, payload):
        request_context = {}
        wire_payload = payload
        if isinstance(payload, dict) and LOCAL_ROI_CONTEXT in payload:
            wire_payload = dict(payload)
            request_context["roi"] = wire_payload.pop(LOCAL_ROI_CONTEXT)
        send_request(self.socket, wire_payload)
        return request_context

    @staticmethod
    def _attach_request_context(result, request_context):
        if (
            isinstance(result, dict)
            and result.get("msg_type") == "PREDICTION"
            and "roi" in request_context
        ):
            result = dict(result)
            result["effective_roi"] = request_context["roi"]
        return result

    def _close_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def _reset_socket(self):
        self._close_socket()
        if self.running:
            self._open_socket()

    def _emit_result_if_current(self, result, timestamp, generation):
        with self._generation_lock:
            if (
                not self.running
                or generation is not self._generation
                or not self._generation_enabled.is_set()
            ):
                return False
            self._last_error = None
        self.result_signal.emit(result, timestamp, generation)
        return True

    def _emit_error_once(self, message, timestamp, generation):
        with self._generation_lock:
            if (
                not self.running
                or generation is not self._generation
                or not self._generation_enabled.is_set()
                or message == self._last_error
            ):
                return False
            self._last_error = message
        self.result_signal.emit(message, timestamp, generation)
        return True

    def stop(self):
        self.running = False
        self._generation_enabled.set()
