import torch
import numpy as np
import os
import sys
from backend.Eventmamba.models.eventmamba_v1 import EventMamba
current_root = os.getcwd()
modules_parent_dir = os.path.join(current_root, "backend", "Eventmamba", "models")

if modules_parent_dir not in sys.path:
    sys.path.append(modules_parent_dir)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class EventMambaPredictor:
    def __init__(self, center_weights, ellipse_weights=None, num_classes=2):
        self.width = 640
        self.height = 480
        self.num_classes = num_classes
        self.center_weights = center_weights
        self.ellipse_weights = ellipse_weights
        self.current_weights = center_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EventMamba(num_classes=num_classes).to(self.device)
        self.load_message = ""

        if not center_weights or not os.path.exists(center_weights):
            raise FileNotFoundError(f"Center 权重文件不存在: {center_weights}")
        if ellipse_weights and not os.path.exists(ellipse_weights):
            raise FileNotFoundError(f"Ellipse 权重文件不存在: {ellipse_weights}")

        self._load_weights(center_weights, num_classes)

    def _load_weights(self, weights_path, num_classes):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            if isinstance(state_dict, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
            self.model.load_state_dict(state_dict)
            self.load_message = f"成功加载权重: {weights_path}"
        except Exception as e:
            raise RuntimeError(f"权重加载失败，请检查路径或文件: {e}") from e
        self.model.eval()
        self.model.apply(inplace_relu)
        dummy_input = torch.randn(1, 3, 1024, device=self.device).float()
        with torch.inference_mode():
            self.model(dummy_input)

    def set_mode(self, mode):
        """切换预测模式并重新加载对应模型"""
        num_classes = 5 if mode == "ellipse" else 2
        weights_path = self.ellipse_weights if mode == "ellipse" else self.center_weights

        if mode == "ellipse" and not self.ellipse_weights:
            print("[EventMamba] 警告：未设置椭圆模式权重文件，保持当前模式")
            return

        if weights_path == self.current_weights and num_classes == self.num_classes:
            return

        self.num_classes = num_classes
        self.current_weights = weights_path
        self.model = EventMamba(num_classes=num_classes).to(self.device)
        self._load_weights(weights_path, num_classes)
        print(f"[EventMamba] 模式切换为 {mode}，num_classes={num_classes}，权重: {weights_path}")

    def process_data(self, data):
        try:
            if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                self.width = data.get("width", 640)
                self.height = data.get("height", 480)
                prediction_mode = data.get("prediction_mode", "center")
                self.set_mode(prediction_mode)
                if self.load_message:
                    return f"{self.load_message}\n相机参数初始化成功\n相机参数: {self.width}x{self.height}\n预测模式: {prediction_mode}"
                return f"相机参数初始化成功\n预测模式: {prediction_mode}"

            is_cropped = True
            if isinstance(data, dict):
                is_cropped = bool(data.get("cropped", True))
                data = data.get("data")

            if data is None:
                return "模型推理出错: 没有收到推理数据"

            data = np.asarray(data, dtype=np.float32)
            if data.ndim != 2:
                return f"模型推理出错: 输入维度应为二维，实际为 {data.shape}"
            if data.shape[1] == 3:
                event_data = data
            elif data.shape[0] == 3:
                event_data = data.T
            else:
                return f"模型推理出错: 输入形状应为 (N, 3)，实际为 {data.shape}"

            event_data = np.ascontiguousarray(event_data, dtype=np.float32)
            data_tensor = torch.from_numpy(event_data).unsqueeze(0).permute(0, 2, 1).to(self.device).float()

            with torch.inference_mode():
                output = self.model(data_tensor)
                result = output.squeeze().cpu().numpy().tolist()
            res_text = f"输出结果为：{result}|cropped:{is_cropped}"
            return res_text

        except Exception as e:
            error_msg = f"模型推理出错: {str(e)}"
            print(error_msg)
            return error_msg
