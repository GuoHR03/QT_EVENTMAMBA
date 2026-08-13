import pickle
import threading

import pytest

import backend.backend_healthcheck as backend_healthcheck
from backend.backend_healthcheck import (
    StartupCancelledError,
    _matching_ready_identity,
    read_backend_log_tail,
    wait_for_backend_ready_with_log,
)
from backend.zmq_protocol import decode_message, encode_message


_PICKLE_EXECUTION_TRIPWIRE = []


def _execute_pickle_tripwire(identity):
    _PICKLE_EXECUTION_TRIPWIRE.append(True)
    return identity


class _MaliciousReady:
    def __init__(self, identity):
        self.identity = identity

    def __reduce__(self):
        return _execute_pickle_tripwire, (self.identity,)


def test_read_backend_log_tail_decodes_and_trims(tmp_path):
    log_path = tmp_path / "backend.log"
    log_path.write_bytes(("a" * 20 + "就绪").encode("utf-8"))

    assert read_backend_log_tail(str(log_path), max_chars=4) == "aa就绪"


def test_read_backend_log_tail_returns_empty_for_missing_file(tmp_path):
    assert read_backend_log_tail(str(tmp_path / "missing.log")) == ""


def test_ready_identity_requires_matching_nonce_and_pid():
    reply = {
        "msg_type": "READY",
        "status": "READY",
        "instance_nonce": "new",
        "pid": 42,
    }

    assert _matching_ready_identity(reply, "new", 42) == reply
    assert _matching_ready_identity(reply, "old", 42) is None
    assert _matching_ready_identity(reply, "new", 43) is None
    assert _matching_ready_identity({**reply, "msg_type": "STATUS"}, "new", 42) is None
    assert _matching_ready_identity("READY", "new", None) is None


@pytest.mark.parametrize("timeout_s", (0, -1, float("inf"), float("nan")))
def test_healthcheck_rejects_non_positive_or_non_finite_timeout(timeout_s):
    with pytest.raises(ValueError, match="finite positive"):
        wait_for_backend_ready_with_log(
            host="127.0.0.1",
            port=5555,
            timeout_s=timeout_s,
            backend_process=None,
            log_path="unused.log",
        )


def test_healthcheck_honors_startup_cancellation(tmp_path):
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(RuntimeError, match="startup cancelled"):
        wait_for_backend_ready_with_log(
            host="127.0.0.1",
            port=5555,
            timeout_s=1,
            backend_process=None,
            log_path=str(tmp_path / "backend.log"),
            cancel_event=cancelled,
        )


def test_cancellation_wins_after_matching_ready_and_preserves_identity(
    monkeypatch,
    tmp_path,
):
    cancelled = threading.Event()
    identity = {
        "msg_type": "READY",
        "status": "READY",
        "instance_nonce": "new",
        "pid": 42,
    }
    requests = []

    class FakeSocket:
        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            pass

        def send_multipart(self, frames):
            requests.append(decode_message(frames))

        def recv_multipart(self):
            cancelled.set()
            return encode_message(identity)

        def close(self, linger=0):
            pass

    class FakeContext:
        def socket(self, socket_type):
            return FakeSocket()

        def term(self):
            pass

    monkeypatch.setattr(backend_healthcheck.zmq, "Context", FakeContext)

    with pytest.raises(StartupCancelledError) as exc_info:
        wait_for_backend_ready_with_log(
            host="127.0.0.1",
            port=5555,
            timeout_s=1,
            backend_process=None,
            log_path=str(tmp_path / "backend.log"),
            expected_nonce="new",
            expected_pid=42,
            cancel_event=cancelled,
        )

    assert exc_info.value.identity == identity
    assert requests == [{"msg_type": "PING", "instance_nonce": "new"}]


def test_malicious_pickle_reply_is_rejected_without_execution_then_ready_retries(
    monkeypatch,
    tmp_path,
):
    identity = {
        "msg_type": "READY",
        "status": "READY",
        "instance_nonce": "new",
        "pid": 42,
    }
    malicious_reply = pickle.dumps(_MaliciousReady(identity))
    replies = [[malicious_reply], encode_message(identity)]
    requests = []
    _PICKLE_EXECUTION_TRIPWIRE.clear()

    class FakeSocket:
        def __init__(self, reply):
            self.reply = reply

        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            pass

        def send_multipart(self, frames):
            requests.append(decode_message(frames))

        def recv_multipart(self):
            return self.reply

        def close(self, linger=0):
            pass

    class FakeContext:
        def socket(self, socket_type):
            return FakeSocket(replies.pop(0))

        def term(self):
            pass

    monkeypatch.setattr(backend_healthcheck.zmq, "Context", FakeContext)

    ready = wait_for_backend_ready_with_log(
        host="127.0.0.1",
        port=5555,
        timeout_s=1,
        poll_interval_s=0.001,
        backend_process=None,
        log_path=str(tmp_path / "backend.log"),
        expected_nonce="new",
        expected_pid=42,
    )

    assert ready == identity
    assert _PICKLE_EXECUTION_TRIPWIRE == []
    assert requests == [
        {"msg_type": "PING", "instance_nonce": "new"},
        {"msg_type": "PING", "instance_nonce": "new"},
    ]
