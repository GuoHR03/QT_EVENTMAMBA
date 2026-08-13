import os

import zmq

from backend.protocol import make_error_response
from backend.zmq_protocol import (
    MAX_TEXT_CHARS,
    ZmqProtocolError,
    configure_socket_limits,
    encode_response,
    receive_request,
)


class ZmqInferenceServer:
    """Platform-neutral REP server used by both inference runtimes."""

    def __init__(
        self,
        model,
        port,
        bind_host,
        *,
        ready_messages=(),
        error_prefix="Inference loop failed",
        instance_nonce=None,
    ):
        self.model = model
        self.port = int(port)
        self.running = True
        self.error_prefix = error_prefix
        self.instance_nonce = instance_nonce
        self.pid = os.getpid()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, 200)
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        configure_socket_limits(self.socket)
        self.socket.bind(f"tcp://{bind_host}:{self.port}")
        for message in ready_messages:
            if message:
                print(message, flush=True)

    def run(self):
        while self.running:
            try:
                data = receive_request(self.socket)
            except ZmqProtocolError as exc:
                print(f"{self.error_prefix}: rejected invalid request: {exc}", flush=True)
                if not self._send_error("Invalid request", code="INVALID_REQUEST"):
                    break
                continue
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                if self.running:
                    print(f"{self.error_prefix}: {exc}", flush=True)
                break

            if data["msg_type"] == "PING":
                if not self._send_response(
                    {
                        "msg_type": "READY",
                        "status": "READY",
                        "instance_nonce": self.instance_nonce,
                        "pid": self.pid,
                    },
                ):
                    break
                continue

            try:
                response = self.model.process_data(data)
            except Exception as exc:
                print(f"{self.error_prefix}: {exc}", flush=True)
                message = str(exc) or "Inference request failed"
                response = make_error_response(
                    message[:MAX_TEXT_CHARS],
                    code="INFERENCE_ERROR",
                )
            if not self._send_response(response):
                break

    def _send_error(self, message, code):
        return self._send_response(make_error_response(message, code=code))

    def _send_response(self, response):
        try:
            frames = encode_response(response)
        except Exception as exc:
            print(f"{self.error_prefix}: invalid response: {exc}", flush=True)
            frames = encode_response(
                make_error_response(
                    "Inference response was invalid",
                    code="INVALID_RESPONSE",
                )
            )
        try:
            self.socket.send_multipart(frames)
            return True
        except zmq.ZMQError as exc:
            if self.running:
                print(f"{self.error_prefix}: reply failed: {exc}", flush=True)
            return False

    def stop(self):
        self.running = False
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
