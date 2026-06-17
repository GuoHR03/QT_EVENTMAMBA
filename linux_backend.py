import argparse
import os
import sys

import torch
import zmq

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "backend"))

from backend.realtime_inference import EventMambaPredictor
from backend.protocol import make_error_response


class InferenceServer:
    def __init__(self, center_weights=None, ellipse_weights=None, initial_mode="center", port=5555):
        self.port = port
        self.center_weights = center_weights
        self.ellipse_weights = ellipse_weights
        self.running = True
        self.current_mode = initial_mode

        current_weights = ellipse_weights if initial_mode == "ellipse" else center_weights
        if not current_weights or not os.path.exists(current_weights):
            print(f"{initial_mode} 模式权重文件不存在: {current_weights}")
            sys.exit(1)

        try:
            self.model = EventMambaPredictor(
                center_weights=center_weights,
                ellipse_weights=ellipse_weights,
                initial_mode=initial_mode,
            )
        except Exception as e:
            print(f"{initial_mode} 模型加载失败: {e}")
            sys.exit(1)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")

    def run(self):
        while self.running:
            try:
                data = self.socket.recv_pyobj()
                if isinstance(data, dict) and data.get("msg_type") == "PING":
                    self.socket.send_string("READY")
                    continue
                result = self.model.process_data(data)
                self.socket.send_pyobj(result)
            except Exception as e:
                print(f"推理循环出错: {e}")
                try:
                    self.socket.send_pyobj(make_error_response(f"Error: {str(e)}"))
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EventMamba Linux Backend Server")
    parser.add_argument("--weights", type=str, help="旧接口，默认按 center 模式权重处理")
    parser.add_argument("--center-weights", type=str, help="Center 模式 .pt(.pth) 权重文件路径")
    parser.add_argument("--ellipse-weights", type=str, help="Ellipse 模式 .pt(.pth) 权重文件路径")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 绑定端口号")
    args = parser.parse_args()

    center_weights = args.center_weights or args.weights
    ellipse_weights = args.ellipse_weights

    if center_weights and ellipse_weights:
        print("错误：请只传入一个模式对应的权重文件")
        sys.exit(1)
    if not center_weights and not ellipse_weights:
        print("错误：必须提供 --center-weights 或 --ellipse-weights 参数")
        sys.exit(1)

    initial_mode = "ellipse" if ellipse_weights else "center"

    server = InferenceServer(
        center_weights=center_weights,
        ellipse_weights=ellipse_weights,
        initial_mode=initial_mode,
        port=args.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
