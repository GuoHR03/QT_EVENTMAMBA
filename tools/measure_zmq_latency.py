import argparse
import statistics
import time

import numpy as np
import zmq

from backend.zmq_protocol import (
    EVENT_SHAPE,
    configure_socket_limits,
    receive_message,
    receive_response,
    send_request,
)


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def print_summary(samples_ms):
    print(f"count: {len(samples_ms)}")
    print(f"min:   {min(samples_ms):.3f} ms")
    print(f"avg:   {statistics.mean(samples_ms):.3f} ms")
    print(f"p50:   {percentile(samples_ms, 50):.3f} ms")
    print(f"p95:   {percentile(samples_ms, 95):.3f} ms")
    print(f"p99:   {percentile(samples_ms, 99):.3f} ms")
    print(f"max:   {max(samples_ms):.3f} ms")


def measure_raw(socket, count, warmup, payload_bytes):
    payload = b"x" * payload_bytes
    samples_ms = []
    for index in range(count + warmup):
        start = time.perf_counter()
        socket.send(payload)
        socket.recv()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= warmup:
            samples_ms.append(elapsed_ms)
    return samples_ms


def measure_wire(socket, count, warmup, points):
    events = np.zeros((points, 3), dtype=np.float32)
    payload = {
        "msg_type": "EVENTS",
        "data": events,
        "timestamp": 0,
        "cropped": True,
    }
    samples_ms = []
    for index in range(count + warmup):
        start = time.perf_counter()
        send_request(socket, payload)
        reply = receive_message(socket)
        if reply.get("msg_type") != "ACK":
            raise RuntimeError(f"unexpected latency server reply: {reply!r}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= warmup:
            samples_ms.append(elapsed_ms)
    return samples_ms


def measure_existing_backend_ping(socket, count, warmup):
    samples_ms = []
    for index in range(count + warmup):
        start = time.perf_counter()
        send_request(socket, {"msg_type": "PING"})
        reply = receive_response(socket)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if reply.get("msg_type") != "READY":
            raise RuntimeError(f"unexpected backend reply: {reply!r}")
        if index >= warmup:
            samples_ms.append(elapsed_ms)
    return samples_ms


def main():
    parser = argparse.ArgumentParser(description="Measure ZMQ round-trip latency.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5566)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--mode",
        choices=("raw", "wire", "backend-ping"),
        default="raw",
    )
    parser.add_argument("--payload-bytes", type=int, default=12288)
    parser.add_argument("--points", type=int, default=1024)
    args = parser.parse_args()
    if args.mode == "wire" and args.points != EVENT_SHAPE[0]:
        parser.error(f"wire mode requires --points {EVENT_SHAPE[0]}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    configure_socket_limits(socket)
    socket.connect(f"tcp://{args.host}:{args.port}")

    try:
        if args.mode == "wire":
            samples_ms = measure_wire(socket, args.count, args.warmup, args.points)
        elif args.mode == "backend-ping":
            samples_ms = measure_existing_backend_ping(socket, args.count, args.warmup)
        else:
            samples_ms = measure_raw(socket, args.count, args.warmup, args.payload_bytes)
        print_summary(samples_ms)
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
