"""Qt/ZMQ transport construction kept outside inference lifecycle policy."""


def create_network_thread(
    frame_queue,
    host,
    port,
    request_timeout_ms,
    result_callback,
    start_paused=False,
):
    from backend.NetworkThread import NetworkThread

    thread = NetworkThread(
        frame_queue,
        host=host,
        port=port,
        request_timeout_ms=request_timeout_ms,
    )
    thread.result_signal.connect(result_callback)
    if start_paused:
        thread.invalidate_generation()
    return thread


def stop_network_thread(thread):
    if thread is None:
        return True
    thread.stop()
    request_interruption = getattr(thread, "requestInterruption", None)
    if callable(request_interruption):
        request_interruption()
    stop_timeout_ms = max(
        1,
        int(getattr(thread, "cooperative_stop_timeout_ms", 2500)),
    )
    if thread.isRunning():
        thread.wait(stop_timeout_ms)
    if thread.isRunning():
        raise RuntimeError(
            "Inference network thread did not stop cooperatively; "
            "the live handle was retained for a safe retry"
        )
    thread.deleteLater()
    return True
