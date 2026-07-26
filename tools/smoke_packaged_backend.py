import argparse
import time

import numpy as np
import zmq


def wait_until_ready(context, host, port, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.connect(f"tcp://{host}:{port}")
        try:
            socket.send_pyobj({"msg_type": "PING"})
            response = socket.recv_string()
            if response == "READY":
                socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
                socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
                return socket
            last_error = RuntimeError(f"Unexpected readiness response: {response!r}")
        except zmq.ZMQError as exc:
            last_error = exc
        socket.close(linger=0)
        time.sleep(0.25)
    raise TimeoutError(f"Backend did not become ready: {last_error}")


def configure_mode(socket, mode):
    socket.send_pyobj(
        {
            "msg_type": "CONFIG",
            "width": 1280,
            "height": 720,
            "prediction_mode": mode,
        }
    )
    response = socket.recv_pyobj()
    if response.get("msg_type") != "STATUS":
        raise RuntimeError(f"{mode} configuration failed: {response}")


def run_prediction(socket, events, mode):
    configure_mode(socket, mode)
    socket.send_pyobj({"data": events, "cropped": True})
    response = socket.recv_pyobj()
    if response.get("msg_type") != "PREDICTION":
        raise RuntimeError(f"{mode} prediction failed: {response}")
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Exercise both models in an already-running packaged backend."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()

    context = zmq.Context()
    socket = None
    try:
        socket = wait_until_ready(
            context,
            args.host,
            args.port,
            args.timeout_s,
        )
        events = np.random.default_rng(7).random((1024, 3), dtype=np.float32)
        center = run_prediction(socket, events, "center")
        ellipse = run_prediction(socket, events, "ellipse")
        print(
            "BACKEND_SMOKE_OK "
            f"center_values={len(center['values'])} "
            f"ellipse_values={len(ellipse['values'])}"
        )
    finally:
        if socket is not None:
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
