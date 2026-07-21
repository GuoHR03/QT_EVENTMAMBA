import argparse
import os
import sys

import zmq


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.protocol import make_error_response
from backend.windows_onnx_predictor import WindowsOnnxPredictorRuntime


class WindowsInferenceServer:
    def __init__(
        self,
        center_model,
        ellipse_model,
        ellipse_matrix,
        custom_op_library,
        initial_mode="center",
        port=5555,
    ):
        self.port = port
        self.running = True
        self.model = WindowsOnnxPredictorRuntime(
            center_model,
            ellipse_model,
            ellipse_matrix,
            custom_op_library,
            initial_mode=initial_mode,
        )
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.bind(f"tcp://127.0.0.1:{self.port}")
        print(self.model.load_message, flush=True)
        print(f"Windows ONNX inference server ready on port {self.port}", flush=True)

    def run(self):
        while self.running:
            try:
                data = self.socket.recv_pyobj()
                if isinstance(data, dict) and data.get("msg_type") == "PING":
                    self.socket.send_string("READY")
                    continue
                self.socket.send_pyobj(self.model.process_data(data))
            except Exception as exc:
                print(f"Windows inference loop failed: {exc}", flush=True)
                try:
                    self.socket.send_pyobj(make_error_response(str(exc)))
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.socket.close(linger=0)
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description="EventMamba Windows ONNX backend")
    parser.add_argument("--center-model", required=True)
    parser.add_argument("--ellipse-model", required=True)
    parser.add_argument("--ellipse-matrix", required=True)
    parser.add_argument("--custom-op-library", required=True)
    parser.add_argument(
        "--initial-mode",
        choices=("center", "ellipse"),
        default="center",
    )
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    server = WindowsInferenceServer(
        center_model=args.center_model,
        ellipse_model=args.ellipse_model,
        ellipse_matrix=args.ellipse_matrix,
        custom_op_library=args.custom_op_library,
        initial_mode=args.initial_mode,
        port=args.port,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
