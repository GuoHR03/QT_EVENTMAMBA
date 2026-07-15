import queue


INFERENCE_STOP_SIGNAL = object()


def enqueue_inference_stop(target_queue, discard_pending):
    """Append a graceful stop or replace pending work with an immediate stop."""
    if not discard_pending:
        target_queue.put(INFERENCE_STOP_SIGNAL)
        return

    while True:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            break

    while True:
        try:
            target_queue.put_nowait(INFERENCE_STOP_SIGNAL)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                continue
