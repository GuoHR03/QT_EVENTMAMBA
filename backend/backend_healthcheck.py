import math
import os
import time

import zmq

from backend.inference_runtime import decode_backend_log
from backend.zmq_protocol import (
    ZmqProtocolError,
    configure_socket_limits,
    receive_response,
    send_request,
)


class StartupCancelledError(RuntimeError):
    """Raised when a caller cancels an in-flight backend startup."""

    def __init__(self, message="Inference service startup cancelled", identity=None):
        super().__init__(message)
        self.identity = identity


def _raise_if_cancelled(cancel_event, identity=None):
    if cancel_event is not None and cancel_event.is_set():
        raise StartupCancelledError(identity=identity)


def wait_for_backend_ready_with_log(
    host,
    port,
    timeout_s,
    backend_process,
    log_path,
    poll_interval_s=0.2,
    expected_nonce=None,
    expected_pid=None,
    cancel_event=None,
):
    timeout_s = _positive_finite(timeout_s, "timeout_s")
    poll_interval_s = _positive_finite(poll_interval_s, "poll_interval_s")
    if expected_pid is not None:
        expected_pid = int(expected_pid)
        if expected_pid <= 0:
            raise ValueError("expected_pid must be a positive integer")

    endpoint = f"tcp://{host}:{port}"
    deadline = time.monotonic() + timeout_s
    context = zmq.Context()
    try:
        while time.monotonic() < deadline:
            _raise_if_cancelled(cancel_event)
            if backend_process is not None and backend_process.poll() is not None:
                details = read_backend_log_tail(log_path)
                if details:
                    raise RuntimeError(f"推理服务启动失败，进程已退出。\n后端日志：\n{details}")
                raise RuntimeError("推理服务启动失败，进程已退出")

            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            remaining_s = max(0.0, deadline - time.monotonic())
            socket_timeout_ms = max(
                1,
                int(min(poll_interval_s, remaining_s) * 1000),
            )
            socket.setsockopt(zmq.RCVTIMEO, socket_timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, socket_timeout_ms)
            configure_socket_limits(socket)
            identity = None
            try:
                socket.connect(endpoint)
                request = {"msg_type": "PING"}
                if expected_nonce is not None:
                    request["instance_nonce"] = expected_nonce
                send_request(socket, request)
                reply = receive_response(socket)
                identity = _matching_ready_identity(
                    reply,
                    expected_nonce=expected_nonce,
                    expected_pid=expected_pid,
                )
            except zmq.Again:
                pass
            except zmq.ZMQError:
                pass
            except ZmqProtocolError:
                # Stale or malformed responders never satisfy readiness.
                pass
            finally:
                socket.close(linger=0)

            # Cancellation wins even if READY and cancel happened in the same
            # request/response turn.  This closes the startup/close race.
            _raise_if_cancelled(cancel_event, identity=identity)
            if identity is not None:
                if expected_nonce is None and expected_pid is None:
                    return None
                return identity

            remaining_s = deadline - time.monotonic()
            if remaining_s > 0:
                delay_s = min(poll_interval_s, remaining_s)
                if cancel_event is None:
                    time.sleep(delay_s)
                elif cancel_event.wait(delay_s):
                    _raise_if_cancelled(cancel_event)

        _raise_if_cancelled(cancel_event)
        details = read_backend_log_tail(log_path)
        if details:
            raise TimeoutError(f"等待推理服务就绪超时。\n后端日志：\n{details}")
        raise TimeoutError("等待推理服务就绪超时")
    finally:
        context.term()


def _positive_finite(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return value


def _matching_ready_identity(reply, expected_nonce=None, expected_pid=None):
    if not isinstance(reply, dict):
        return None
    if reply.get("msg_type") != "READY" or reply.get("status") != "READY":
        return None
    if expected_nonce is not None and reply.get("instance_nonce") != expected_nonce:
        return None
    try:
        pid = int(reply.get("pid"))
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if expected_pid is not None and pid != expected_pid:
        return None
    return {
        "msg_type": "READY",
        "status": "READY",
        "instance_nonce": reply.get("instance_nonce"),
        "pid": pid,
    }


def read_backend_log_tail(log_path, max_chars=4000):
    try:
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "rb") as handle:
            raw = handle.read()
        content = decode_backend_log(raw)
        return content[-max_chars:].strip()
    except Exception:
        return ""
