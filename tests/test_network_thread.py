import pickle
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from backend.NetworkThread import NetworkThread
from backend.protocol import LOCAL_ROI_CONTEXT
from backend.zmq_protocol import ZmqProtocolError, decode_message, encode_message


def _write_tripwire(path):
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousPickle:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _write_tripwire, (self.path,)


class FakeWireSocket:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(list(frames))

    def recv_multipart(self):
        return next(self.replies)


def test_network_generation_drops_old_requests_and_prioritizes_replacement():
    input_queue = queue.Queue()
    input_queue.put_nowait({"msg_type": "EVENTS", "timestamp": 1})
    thread = NetworkThread(input_queue)

    old_payload, old_generation = thread._get_latest_payload_for_generation()
    assert old_payload["timestamp"] == 1

    new_generation = thread.invalidate_generation()
    assert new_generation != old_generation
    assert not thread._is_current_generation(old_generation)

    config = {"msg_type": "CONFIG", "width": 320, "height": 240}
    thread.replace_pending_payload(config)
    input_queue.put_nowait({"msg_type": "EVENTS", "timestamp": 2})
    thread.resume_generation()

    payload, generation = thread._get_latest_payload_for_generation()
    assert payload == config
    assert generation == new_generation
    payload, generation = thread._get_latest_payload_for_generation()
    assert payload["timestamp"] == 2
    assert generation == new_generation
    thread.stop()


def test_invalidating_generation_discards_old_priority_payload():
    thread = NetworkThread(queue.Queue())
    thread.replace_pending_payload({"msg_type": "CONFIG", "width": 640})

    thread.invalidate_generation()

    assert thread._priority_payload is None
    thread.stop()


def test_invalidating_generation_discards_only_pre_barrier_queue_items():
    input_queue = queue.Queue()
    input_queue.put_nowait({"msg_type": "EVENTS", "timestamp": 1})
    thread = NetworkThread(input_queue)

    generation = thread.invalidate_generation()
    input_queue.put_nowait({"msg_type": "EVENTS", "timestamp": 2})
    thread.resume_generation()

    payload, payload_generation = thread._get_latest_payload_for_generation()
    assert payload["timestamp"] == 2
    assert payload_generation == generation
    thread.stop()


def test_waiting_for_input_does_not_hold_generation_lock():
    entered_get = threading.Event()
    release_get = threading.Event()

    class BlockingQueue:
        def get(self, timeout=None):
            entered_get.set()
            release_get.wait(timeout=1)
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    thread = NetworkThread(BlockingQueue())
    reader = threading.Thread(target=thread._get_latest_payload_for_generation)
    invalidated = threading.Event()
    invalidator = threading.Thread(
        target=lambda: (thread.invalidate_generation(), invalidated.set())
    )
    reader.start()
    assert entered_get.wait(timeout=1)
    invalidator.start()
    try:
        assert invalidated.wait(timeout=0.1)
    finally:
        release_get.set()
        reader.join(timeout=1)
        invalidator.join(timeout=1)
        thread.stop()


def test_network_thread_encodes_config_and_events_without_object_serialization():
    thread = NetworkThread(queue.Queue())
    socket = FakeWireSocket()
    thread.socket = socket

    thread._send_payload(
        {
            "msg_type": "CONFIG",
            "width": 640,
            "height": 480,
            "prediction_mode": "center",
        }
    )
    thread._send_payload(
        {
            "msg_type": "EVENTS",
            "data": np.zeros((1024, 3), dtype=np.float32),
            "timestamp": 7,
            "cropped": True,
        }
    )

    assert len(socket.sent[0]) == 1
    assert decode_message(socket.sent[0])["msg_type"] == "CONFIG"
    assert len(socket.sent[1]) == 2
    decoded_events = decode_message(socket.sent[1])
    assert decoded_events["timestamp"] == 7
    assert decoded_events["data"].shape == (1024, 3)
    thread.stop()


def test_network_thread_strictly_decodes_valid_response():
    response = {
        "msg_type": "PREDICTION",
        "values": [0.25, 0.75],
        "cropped": True,
        "mode": "center",
    }
    thread = NetworkThread(queue.Queue())
    thread.socket = FakeWireSocket([encode_message(response)])

    assert thread._recv_result() == response
    thread.stop()


def test_network_thread_keeps_effective_roi_as_local_response_context():
    thread = NetworkThread(queue.Queue())
    socket = FakeWireSocket()
    thread.socket = socket
    payload = {
        "msg_type": "EVENTS",
        "data": np.zeros((1024, 3), dtype=np.float32),
        "timestamp": 7,
        "cropped": True,
        LOCAL_ROI_CONTEXT: (10, 20, 30, 40),
    }

    context = thread._send_payload(payload)
    sent = decode_message(socket.sent[0])
    result = thread._attach_request_context(
        {
            "msg_type": "PREDICTION",
            "values": [0.5, 0.5],
            "cropped": True,
            "mode": "center",
        },
        context,
    )

    assert LOCAL_ROI_CONTEXT not in sent
    assert result["effective_roi"] == (10, 20, 30, 40)
    thread.stop()


def test_network_thread_rejects_malicious_pickle_response_without_execution(tmp_path):
    marker = tmp_path / "network-pickle-executed"
    malicious = pickle.dumps(_MaliciousPickle(marker))
    thread = NetworkThread(queue.Queue())
    thread.socket = FakeWireSocket([[malicious]])

    with pytest.raises(ZmqProtocolError):
        thread._recv_result()

    assert not marker.exists()
    thread.stop()
