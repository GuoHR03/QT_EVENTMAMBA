import argparse
import os

import torch

from backend.inference_server import ZmqInferenceServer
from backend.realtime_inference import EventMambaPredictor


class InferenceServer(ZmqInferenceServer):
    def __init__(
        self,
        center_weights=None,
        ellipse_weights=None,
        initial_mode="center",
        port=5555,
        instance_nonce=None,
    ):
        current_weights = (
            ellipse_weights if initial_mode == "ellipse" else center_weights
        )
        if not current_weights or not os.path.exists(current_weights):
            raise FileNotFoundError(
                f"{initial_mode} mode weights do not exist: {current_weights}"
            )

        model = EventMambaPredictor(
            center_weights=center_weights,
            ellipse_weights=ellipse_weights,
            initial_mode=initial_mode,
        )
        super().__init__(
            model,
            port,
            "127.0.0.1",
            error_prefix="Inference loop failed",
            instance_nonce=instance_nonce,
        )

    def stop(self):
        super().stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args():
    parser = argparse.ArgumentParser(description="EventMamba Linux backend server")
    parser.add_argument(
        "--weights",
        help="Legacy alias for center-mode weights",
    )
    parser.add_argument("--center-weights", help="Center-mode .pt/.pth weights")
    parser.add_argument("--ellipse-weights", help="Ellipse-mode .pt/.pth weights")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--instance-nonce")
    return parser.parse_args()


def main():
    args = _parse_args()
    center_weights = args.center_weights or args.weights
    ellipse_weights = args.ellipse_weights

    if center_weights and ellipse_weights:
        raise SystemExit("Provide weights for only one initial prediction mode")
    if not center_weights and not ellipse_weights:
        raise SystemExit("--center-weights or --ellipse-weights is required")

    initial_mode = "ellipse" if ellipse_weights else "center"
    try:
        server = InferenceServer(
            center_weights=center_weights,
            ellipse_weights=ellipse_weights,
            initial_mode=initial_mode,
            port=args.port,
            instance_nonce=args.instance_nonce,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to start {initial_mode} model: {exc}") from exc

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
