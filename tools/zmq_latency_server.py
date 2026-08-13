import argparse

import zmq

from backend.zmq_protocol import (
    configure_socket_limits,
    receive_message,
    send_message,
    validate_request_message,
)


def run_server(port, mode, bind_host="127.0.0.1"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    configure_socket_limits(socket)
    socket.bind(f"tcp://{bind_host}:{port}")
    print(
        f"ZMQ latency server listening on tcp://{bind_host}:{port} ({mode})",
        flush=True,
    )

    try:
        while True:
            if mode == "wire":
                payload = receive_message(socket)
                if isinstance(payload, dict) and payload.get("msg_type") == "STOP":
                    send_message(socket, {"msg_type": "STOPPED"})
                    break
                validate_request_message(payload)
                send_message(socket, {"msg_type": "ACK"})
            else:
                payload = socket.recv()
                if payload == b"STOP":
                    socket.send(b"STOPPED")
                    break
                socket.send(b"ACK")
    finally:
        socket.close(linger=0)
        context.term()


def main():
    parser = argparse.ArgumentParser(description="Minimal ZMQ REP server for Windows <-> WSL latency tests.")
    parser.add_argument("--port", type=int, default=5566)
    parser.add_argument("--mode", choices=("raw", "wire"), default="raw")
    parser.add_argument("--bind-host", default="127.0.0.1")
    args = parser.parse_args()
    run_server(args.port, args.mode, bind_host=args.bind_host)


if __name__ == "__main__":
    main()
