import queue

from backend.inference_worker_control import INFERENCE_STOP_SIGNAL, enqueue_inference_stop


def test_graceful_inference_stop_preserves_pending_windows():
    target_queue = queue.Queue(maxsize=3)
    target_queue.put_nowait("first")
    target_queue.put_nowait("second")

    enqueue_inference_stop(target_queue, discard_pending=False)

    assert target_queue.get_nowait() == "first"
    assert target_queue.get_nowait() == "second"
    assert target_queue.get_nowait() is INFERENCE_STOP_SIGNAL


def test_immediate_inference_stop_discards_pending_windows():
    target_queue = queue.Queue(maxsize=2)
    target_queue.put_nowait("first")
    target_queue.put_nowait("second")

    enqueue_inference_stop(target_queue, discard_pending=True)

    assert target_queue.get_nowait() is INFERENCE_STOP_SIGNAL
    assert target_queue.empty()
