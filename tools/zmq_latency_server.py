import argparse
import pickle

import zmq


def run_server(port, mode):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://0.0.0.0:{port}")
    print(f"ZMQ latency server listening on tcp://0.0.0.0:{port} ({mode})", flush=True)

    try:
        while True:
            if mode in ("pyobj", "pyobj-list"):
                payload = socket.recv_pyobj()
                if isinstance(payload, dict) and payload.get("msg_type") == "STOP":
                    socket.send_pyobj({"msg_type": "STOPPED"})
                    break
                socket.send_pyobj({"msg_type": "ACK"})
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
    parser.add_argument("--mode", choices=("raw", "pyobj", "pyobj-list"), default="raw")
    args = parser.parse_args()
    run_server(args.port, args.mode)


if __name__ == "__main__":
    main()
