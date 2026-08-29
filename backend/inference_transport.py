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
    if thread.isRunning() and not thread.wait(2000):
        thread.terminate()
        thread.wait(500)
    if thread.isRunning():
        raise RuntimeError("Inference network thread did not stop")
    thread.deleteLater()
    return True
