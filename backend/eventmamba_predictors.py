import torch

from backend.model_assets import MODE_CENTER, MODE_ELLIPSE, validate_model_asset
from backend.models import vsa
from backend.models.eventmamba_v1 import EventMamba as CenterEventMamba
from backend.models.eventmamba_v3 import EventMamba as EllipseEventMamba


def inplace_relu(module):
    if "ReLU" in module.__class__.__name__:
        module.inplace = True


class BasePredictor:
    def __init__(self, weights_path, device):
        self.weights_path = weights_path
        self.device = device
        self.load_message = ""

    @staticmethod
    def extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    def predict(self, event_data):
        raise NotImplementedError


class CenterPredictor(BasePredictor):
    def __init__(self, weights_path, device):
        validate_model_asset(MODE_CENTER, weights_path)
        super().__init__(weights_path, device)
        self.model = CenterEventMamba(num_classes=2).to(self.device)
        self._load_weights()

    def _load_weights(self):
        try:
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(self.extract_state_dict(state_dict))
            self.load_message = f"成功加载中心点权重: {self.weights_path}"
        except Exception as exc:
            raise RuntimeError(f"中心点权重加载失败，请检查路径或文件: {exc}") from exc

        self.model.eval()
        self.model.apply(inplace_relu)
        dummy_input = torch.randn(1, 3, 1024, device=self.device).float()
        with torch.inference_mode():
            self.model(dummy_input)

    def predict(self, event_data):
        data_tensor = (
            torch.from_numpy(event_data)
            .unsqueeze(0)
            .permute(0, 2, 1)
            .to(self.device)
            .float()
        )
        with torch.inference_mode():
            output = self.model(data_tensor)
        return output.squeeze().cpu().numpy().tolist()


class EllipsePredictor(BasePredictor):
    def __init__(self, weights_path, device):
        asset = validate_model_asset(MODE_ELLIPSE, weights_path)
        super().__init__(weights_path, device)
        self.matrix_path = asset.matrix_path
        self.model = EllipseEventMamba(num_classes=1024).to(self.device)
        self.matrix_A = None
        self._load_runtime_assets()

    def _load_runtime_assets(self):
        try:
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(self.extract_state_dict(state_dict))
            matrix_a = torch.load(self.matrix_path, map_location=self.device)
            self.matrix_A = matrix_a.to(self.device).float()
            self.load_message = (
                f"成功加载椭圆权重: {self.weights_path}; "
                f"成功加载 matrix_A: {self.matrix_path}"
            )
        except Exception as exc:
            raise RuntimeError(f"椭圆模型加载失败，请检查权重或 matrix_A: {exc}") from exc

        self.model.eval()
        self.model.apply(inplace_relu)
        dummy_input = torch.randn(1, 3, 1024, device=self.device).float()
        with torch.inference_mode():
            self.model(dummy_input)

    def predict(self, event_data):
        data_tensor = (
            torch.from_numpy(event_data)
            .unsqueeze(0)
            .permute(0, 2, 1)
            .to(self.device)
            .float()
        )
        with torch.inference_mode():
            predict_vector = self.model(data_tensor)
            real_part, imag_part = torch.chunk(predict_vector, chunks=2, dim=-1)
            predict_vsa = torch.complex(real_part, imag_part)
            decoded = vsa.Decode_VSA(predict_vsa, self.matrix_A, isELL=True)
        return decoded.squeeze(0).cpu().numpy().tolist()
