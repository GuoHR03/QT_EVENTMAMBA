import pickle
import socket as tcp_socket
import threading
from pathlib import Path

import numpy as np
import pytest
import zmq

from backend import inference_server
from backend.protocol import make_prediction_response, make_status_response
from backend.zmq_protocol import (
    EVENT_DTYPE,
    EVENT_SHAPE,
    configure_socket_limits,
    decode_message,
    encode_message,
    receive_response,
    send_request,
)


def _write_server_tripwire(path):
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousServerRequest:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _write_server_tripwire, (self.path,)


class FakeSocket:
    def __init__(self, requests):
        self.requests = iter(requests)
        self.options = []
        self.bound_endpoint = None
        self.sent = []
        self.closed = False

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def bind(self, endpoint):
        self.bound_endpoint = endpoint

    def recv_multipart(self):
        request = next(self.requests)
        if isinstance(request, BaseException):
            raise request
        return request

    def send_multipart(self, frames):
        self.sent.append(decode_message(frames))

    def close(self, linger=0):
        self.closed = linger == 0


class FakeContext:
    def __init__(self, socket):
        self._socket = socket
        self.terminated = False

    def socket(self, _kind):
        return self._socket

    def term(self):
        self.terminated = True


class FakeModel:
    def __init__(self, error=None):
        self.error = error
        self.requests = []

    def process_data(self, data):
        self.requests.append(data)
        if self.error is not None:
            raise self.error
        if data["msg_type"] == "CONFIG":
            return make_status_response(
                "Configuration updated",
                width=data["width"],
                height=data["height"],
                mode=data["prediction_mode"],
            )
        return make_prediction_response(
            [0.25, 0.75],
            cropped=data["cropped"],
            mode="center",
        )


def _run_server(monkeypatch, requests, model=None, **server_kwargs):
    socket = FakeSocket([*requests, KeyboardInterrupt()])
    context = FakeContext(socket)
    monkeypatch.setattr(inference_server.zmq, "Context", lambda: context)
    server = inference_server.ZmqInferenceServer(
        model or FakeModel(),
        6000,
        "127.0.0.1",
        **server_kwargs,
    )

    with pytest.raises(KeyboardInterrupt):
        server.run()

    return server, socket, context


def test_inference_server_shares_ping_and_request_loop(monkeypatch):
    events = np.zeros(EVENT_SHAPE, dtype=EVENT_DTYPE)
    requests = [
        encode_message({"msg_type": "PING"}),
        encode_message(
            {
                "msg_type": "CONFIG",
                "width": 640,
                "height": 480,
                "prediction_mode": "center",
            }
        ),
        encode_message(
            {
                "msg_type": "EVENTS",
                "data": events,
                "timestamp": 123,
                "cropped": True,
            }
        ),
    ]
    model = FakeModel()

    server, socket, context = _run_server(monkeypatch, requests, model=model)

    assert socket.bound_endpoint == "tcp://127.0.0.1:6000"
    assert all(option != inference_server.zmq.CONFLATE for option, _ in socket.options)
    assert socket.sent == [
        {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": None,
            "pid": server.pid,
        },
        {
            "msg_type": "STATUS",
            "message": "Configuration updated",
            "width": 640,
            "height": 480,
            "mode": "center",
        },
        {
            "msg_type": "PREDICTION",
            "values": [0.25, 0.75],
            "cropped": True,
            "mode": "center",
        },
    ]
    assert [request["msg_type"] for request in model.requests] == ["CONFIG", "EVENTS"]

    server.stop()
    assert socket.closed
    assert context.terminated
    assert server.socket is None
    assert server.context is None


def test_inference_server_nonce_handshake_identifies_exact_process(monkeypatch):
    server, socket, _ = _run_server(
        monkeypatch,
        [encode_message({"msg_type": "PING", "instance_nonce": "requested"})],
        instance_nonce="owned",
    )

    assert socket.sent == [
        {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": "owned",
            "pid": server.pid,
        }
    ]


def test_invalid_json_gets_safe_error_and_next_ping_still_succeeds(monkeypatch):
    server, socket, _ = _run_server(
        monkeypatch,
        [[b"{not-json"], encode_message({"msg_type": "PING"})],
    )

    assert socket.sent == [
        {
            "msg_type": "ERROR",
            "message": "Invalid request",
            "code": "INVALID_REQUEST",
        },
        {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": None,
            "pid": server.pid,
        },
    ]


def test_model_exception_gets_error_and_next_ping_still_succeeds(monkeypatch):
    model = FakeModel(error=RuntimeError("model failed"))
    requests = [
        encode_message(
            {
                "msg_type": "CONFIG",
                "width": 640,
                "height": 480,
                "prediction_mode": "center",
            }
        ),
        encode_message({"msg_type": "PING"}),
    ]

    server, socket, _ = _run_server(monkeypatch, requests, model=model)

    assert socket.sent == [
        {
            "msg_type": "ERROR",
            "message": "model failed",
            "code": "INFERENCE_ERROR",
        },
        {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": None,
            "pid": server.pid,
        },
    ]


def test_real_zmq_server_rejects_pickle_then_keeps_serving(tmp_path):
    marker = tmp_path / "server-pickle-executed"
    malicious = pickle.dumps(_MaliciousServerRequest(marker))
    with tcp_socket.socket(tcp_socket.AF_INET, tcp_socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = inference_server.ZmqInferenceServer(
        FakeModel(),
        port,
        "127.0.0.1",
        instance_nonce="owned",
    )
    worker = threading.Thread(target=server.run, daemon=True)
    worker.start()

    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.LINGER, 0)
    client.setsockopt(zmq.RCVTIMEO, 2000)
    client.setsockopt(zmq.SNDTIMEO, 2000)
    configure_socket_limits(client)
    client.connect(f"tcp://127.0.0.1:{port}")
    try:
        client.send_multipart([malicious])
        error = receive_response(client)
        assert error == {
            "msg_type": "ERROR",
            "message": "Invalid request",
            "code": "INVALID_REQUEST",
        }
        assert not marker.exists()

        send_request(client, {"msg_type": "PING", "instance_nonce": "requested"})
        ready = receive_response(client)
        assert ready == {
            "msg_type": "READY",
            "status": "READY",
            "instance_nonce": "owned",
            "pid": server.pid,
        }
    finally:
        server.running = False
        worker.join(timeout=2)
        client.close(linger=0)
        context.term()
        server.stop()

    assert not worker.is_alive()
