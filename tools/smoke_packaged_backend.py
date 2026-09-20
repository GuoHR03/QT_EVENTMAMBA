import argparse
import time

import numpy as np
import zmq

from backend.zmq_protocol import (
    ZmqProtocolError,
    configure_socket_limits,
    receive_response,
    send_request,
)


def wait_until_ready(context, host, port, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        configure_socket_limits(socket)
        socket.connect(f"tcp://{host}:{port}")
        try:
            send_request(socket, {"msg_type": "PING"})
            response = receive_response(socket)
            if response.get("msg_type") == "READY":
                socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
                socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
                return socket
            last_error = RuntimeError(f"Unexpected readiness response: {response!r}")
        except (zmq.ZMQError, ZmqProtocolError) as exc:
            last_error = exc
        socket.close(linger=0)
        time.sleep(0.25)
    raise TimeoutError(f"Backend did not become ready: {last_error}")


def configure_mode(socket, mode):
    send_request(
        socket,
        {
            "msg_type": "CONFIG",
            "width": 1280,
            "height": 720,
            "prediction_mode": mode,
        }
    )
    response = receive_response(socket)
    if response.get("msg_type") != "STATUS":
        raise RuntimeError(f"{mode} configuration failed: {response}")


def run_prediction(socket, events, mode):
    configure_mode(socket, mode)
    return request_prediction(socket, events, mode)


def request_prediction(socket, events, mode):
    send_request(
        socket,
        {
            "msg_type": "EVENTS",
            "data": events,
            "timestamp": 0,
            "cropped": True,
        },
    )
    response = receive_response(socket)
    if response.get("msg_type") != "PREDICTION":
        raise RuntimeError(f"{mode} prediction failed: {response}")
    return response


def benchmark_mode(socket, events, mode, repeats):
    configure_mode(socket, mode)
    for _ in range(5):
        request_prediction(socket, events, mode)
    roundtrip_times = []
    inference_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        response = request_prediction(socket, events, mode)
        roundtrip_times.append((time.perf_counter() - started) * 1000.0)
        inference_times.append(float(response["inference_ms"]))
    other_times = np.subtract(roundtrip_times, inference_times)
    return {
        "roundtrip_mean_ms": float(np.mean(roundtrip_times)),
        "roundtrip_p50_ms": float(np.percentile(roundtrip_times, 50)),
        "roundtrip_p95_ms": float(np.percentile(roundtrip_times, 95)),
        "inference_mean_ms": float(np.mean(inference_times)),
        "inference_p50_ms": float(np.percentile(inference_times, 50)),
        "inference_p95_ms": float(np.percentile(inference_times, 95)),
        "other_mean_ms": float(np.mean(other_times)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exercise both models in an already-running packaged backend."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark_repeats < 0:
        parser.error("--benchmark-repeats must be non-negative")

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
        center_again = run_prediction(socket, events, "center")
        for mode, response, expected_values in (
            ("center", center, 2),
            ("ellipse", ellipse, 5),
            ("center-after-switch", center_again, 2),
        ):
            values = np.asarray(response["values"], dtype=np.float64)
            if values.shape != (expected_values,) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid {mode} prediction: {response}")
        print(
            "BACKEND_SMOKE_OK "
            f"center_values={len(center['values'])} "
            f"ellipse_values={len(ellipse['values'])} "
            f"center_after_switch_values={len(center_again['values'])}"
        )
        if args.benchmark_repeats:
            center_benchmark = benchmark_mode(
                socket, events, "center", args.benchmark_repeats
            )
            ellipse_benchmark = benchmark_mode(
                socket, events, "ellipse", args.benchmark_repeats
            )
            print(
                "BACKEND_BENCHMARK "
                f"repeats={args.benchmark_repeats} "
                f"center_roundtrip_p50_ms={center_benchmark['roundtrip_p50_ms']:.3f} "
                f"center_roundtrip_p95_ms={center_benchmark['roundtrip_p95_ms']:.3f} "
                f"center_inference_p50_ms={center_benchmark['inference_p50_ms']:.3f} "
                f"center_inference_p95_ms={center_benchmark['inference_p95_ms']:.3f} "
                f"ellipse_roundtrip_p50_ms={ellipse_benchmark['roundtrip_p50_ms']:.3f} "
                f"ellipse_roundtrip_p95_ms={ellipse_benchmark['roundtrip_p95_ms']:.3f} "
                f"ellipse_inference_p50_ms={ellipse_benchmark['inference_p50_ms']:.3f} "
                f"ellipse_inference_p95_ms={ellipse_benchmark['inference_p95_ms']:.3f}"
            )
    finally:
        if socket is not None:
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
