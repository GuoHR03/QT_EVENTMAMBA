import os

import numpy as np
import torch

from backend.models.eventmamba_v1 import EventMamba


def inplace_relu(module):
    classname = module.__class__.__name__
    if classname.find("ReLU") != -1:
        module.inplace = True


class BasePredictor:
    def __init__(self, weights_path, device):
        self.weights_path = weights_path
        self.device = device
        self.load_message = ""

        if not weights_path:
            raise FileNotFoundError("未提供权重文件")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    def predict(self, event_data):
        raise NotImplementedError


class CenterPredictor(BasePredictor):
    def __init__(self, weights_path, device):
        super().__init__(weights_path, device)
        self.model = EventMamba(num_classes=2).to(self.device)
        self._load_weights()

    def _load_weights(self):
        try:
            state_dict = torch.load(self.weights_path, map_location=self.device)
            if isinstance(state_dict, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
            self.model.load_state_dict(state_dict)
            self.load_message = f"成功加载中心点权重: {self.weights_path}"
        except Exception as e:
            raise RuntimeError(f"中心点权重加载失败，请检查路径或文件: {e}") from e

        self.model.eval()
        self.model.apply(inplace_relu)
        dummy_input = torch.randn(1, 3, 1024, device=self.device).float()
        with torch.inference_mode():
            self.model(dummy_input)

    def predict(self, event_data):
        data_tensor = torch.from_numpy(event_data).unsqueeze(0).permute(0, 2, 1).to(self.device).float()
        with torch.inference_mode():
            output = self.model(data_tensor)
        return output.squeeze().cpu().numpy().tolist()


class EllipsePredictor(BasePredictor):
    def __init__(self, weights_path, device):
        super().__init__(weights_path, device)
        self.load_message = (
            f"已收到椭圆权重路径: {self.weights_path}；"
            "椭圆网络框架接口已预留，但具体模型尚未接入"
        )

    def predict(self, event_data):
        raise NotImplementedError("椭圆网络框架尚未实现，请在 EllipsePredictor 中接入实际模型")


class EventMambaPredictor:
    def __init__(self, center_weights=None, ellipse_weights=None, initial_mode="center"):
        self.width = 640
        self.height = 480
        self.center_weights = center_weights
        self.ellipse_weights = ellipse_weights
        self.current_mode = initial_mode
        self.current_weights = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.load_message = ""
        self._build_predictor(initial_mode)

    def _build_predictor(self, mode):
        weights_path = self.ellipse_weights if mode == "ellipse" else self.center_weights
        if mode == "ellipse":
            self.predictor = EllipsePredictor(weights_path, self.device)
        else:
            self.predictor = CenterPredictor(weights_path, self.device)
        self.current_mode = mode
        self.current_weights = weights_path
        self.load_message = self.predictor.load_message

    def set_mode(self, mode):
        """切换预测模式，并加载该模式对应的网络框架。"""
        weights_path = self.ellipse_weights if mode == "ellipse" else self.center_weights
        if not weights_path:
            print(f"[Inference] 警告：未设置 {mode} 模式权重文件，保持当前模式")
            return

        if mode == self.current_mode and weights_path == self.current_weights:
            return

        self._build_predictor(mode)
        print(f"[Inference] 模式切换为 {mode}，权重: {weights_path}")

    def _parse_event_data(self, data):
        if data is None:
            raise ValueError("没有收到推理数据")

        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"输入维度应为二维，实际为 {data.shape}")
        if data.shape[1] == 3:
            event_data = data
        elif data.shape[0] == 3:
            event_data = data.T
        else:
            raise ValueError(f"输入形状应为 (N, 3)，实际为 {data.shape}")

        return np.ascontiguousarray(event_data, dtype=np.float32)

    def process_data(self, data):
        try:
            if isinstance(data, dict) and data.get("msg_type") == "CONFIG":
                self.width = data.get("width", 640)
                self.height = data.get("height", 480)
                prediction_mode = data.get("prediction_mode", self.current_mode)
                self.set_mode(prediction_mode)
                if self.load_message:
                    return (
                        f"{self.load_message}\n"
                        f"相机参数初始化成功\n"
                        f"相机参数: {self.width}x{self.height}\n"
                        f"预测模式: {prediction_mode}"
                    )
                return f"相机参数初始化成功\n预测模式: {prediction_mode}"

            is_cropped = True
            if isinstance(data, dict):
                is_cropped = bool(data.get("cropped", True))
                data = data.get("data")

            event_data = self._parse_event_data(data)
            result = self.predictor.predict(event_data)
            return f"输出结果为：{result}|cropped:{is_cropped}"

        except Exception as e:
            error_msg = f"模型推理出错: {str(e)}"
            print(error_msg)
            return error_msg
