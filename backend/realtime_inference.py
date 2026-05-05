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
    def __init__(self, weights_path):
        self.width = 640
        self.height = 480
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EventMamba(num_classes=2).to(self.device)
        self.load_message = ""

        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

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

    def process_data(self, data):
        try:
            if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                self.width = data["width"]
                self.height = data["height"]
                if self.load_message:
                    return f"{self.load_message}\n相机参数初始化成功\n相机参数: {self.width}x{self.height}"
                return "相机参数初始化成功"

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
