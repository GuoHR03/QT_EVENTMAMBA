import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "backend"))
sys.path.append(os.path.join(current_dir, "backend", "Eventmamba"))
sys.path.append(os.path.join(current_dir, "backend", "Eventmamba", "models"))
import argparse
import zmq
import torch
from backend.realtime_inference import EventMambaPredictor

class InferenceServer:
    def __init__(self, center_weights, ellipse_weights=None, port=5555):
        self.port = port
        self.center_weights = center_weights
        self.ellipse_weights = ellipse_weights
        self.running = True
        self.current_mode = "center"

        if not center_weights or not os.path.exists(center_weights):
            print(f"Center 权重文件不存在: {center_weights}")
            sys.exit(1)

        try:
            self.model = EventMambaPredictor(center_weights, ellipse_weights, num_classes=2)
            self.current_mode = "center"
        except Exception as e:
            print(f"Center 模型加载失败: {e}")
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
                if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                    result_text = self.model.process_data(data)
                    self.socket.send_string(result_text)
                    continue

                result_text = self.model.process_data(data)
                self.socket.send_string(result_text)

            except Exception as e:
                print(f"推理循环出错: {e}")
                try:
                    self.socket.send_string(f"Error: {str(e)}")
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
    parser.add_argument("--weights", type=str, help="Center 模式 .pt(.pth) 权重文件的路径（兼容旧接口）")
    parser.add_argument("--center-weights", type=str, help="Center 模式 .pt(.pth) 权重文件的路径")
    parser.add_argument("--ellipse-weights", type=str, help="Ellipse 模式 .pt(.pth) 权重文件的路径")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 绑定的端口号")
    args = parser.parse_args()

    center_weights = args.center_weights or args.weights
    if not center_weights:
        print("错误：必须提供 --weights 或 --center-weights 参数")
        sys.exit(1)

    ellipse_weights = args.ellipse_weights
    if not ellipse_weights:
        print("警告：未提供 --ellipse-weights，将无法切换到椭圆模式")

    server = InferenceServer(
        center_weights=center_weights,
        ellipse_weights=ellipse_weights,
        port=args.port
    )

    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
