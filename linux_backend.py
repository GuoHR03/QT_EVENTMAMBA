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
    def __init__(self, weight_path, port=5555):
        self.port = port
        self.weight_path = weight_path
        self.running = True

        try:
            self.model = EventMambaPredictor(weight_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
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
    parser.add_argument("--weights", type=str, required=True, help="模型 .pt(.pth) 权重文件的路径")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 绑定的端口号")
    args = parser.parse_args()
    server = InferenceServer(weight_path=args.weights, port=args.port)
    
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
