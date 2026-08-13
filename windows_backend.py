import argparse

from backend.inference_server import ZmqInferenceServer
from backend.windows_onnx_predictor import WindowsOnnxPredictorRuntime


class WindowsInferenceServer(ZmqInferenceServer):
    def __init__(
        self,
        center_model,
        ellipse_model,
        ellipse_matrix,
        custom_op_library,
        initial_mode="center",
        port=5555,
        instance_nonce=None,
    ):
        model = WindowsOnnxPredictorRuntime(
            center_model,
            ellipse_model,
            ellipse_matrix,
            custom_op_library,
            initial_mode=initial_mode,
        )
        super().__init__(
            model,
            port,
            "127.0.0.1",
            ready_messages=(
                model.load_message,
                f"Windows ONNX inference server ready on port {port}",
            ),
            error_prefix="Windows inference loop failed",
            instance_nonce=instance_nonce,
        )


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
    parser.add_argument("--instance-nonce")
    args = parser.parse_args()
    server = WindowsInferenceServer(
        center_model=args.center_model,
        ellipse_model=args.ellipse_model,
        ellipse_matrix=args.ellipse_matrix,
        custom_op_library=args.custom_op_library,
        initial_mode=args.initial_mode,
        port=args.port,
        instance_nonce=args.instance_nonce,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
