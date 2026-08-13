import os
from dataclasses import dataclass


MODE_CENTER = "center"
MODE_ELLIPSE = "ellipse"
MATRIX_A_FILENAME = "matrix_A.pt"


@dataclass(frozen=True)
class ModelAsset:
    mode: str
    weights_path: str
    matrix_path: str | None = None


def matrix_path_for_weights(weights_path):
    if not weights_path:
        return None
    return os.path.join(os.path.dirname(weights_path), MATRIX_A_FILENAME)


def validate_model_asset(mode, weights_path, exists=os.path.exists):
    if mode not in (MODE_CENTER, MODE_ELLIPSE):
        raise ValueError(f"Unsupported prediction mode: {mode}")
    if not weights_path:
        raise FileNotFoundError("未提供权重文件")
    if not exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    matrix_path = None
    if mode == MODE_ELLIPSE:
        matrix_path = matrix_path_for_weights(weights_path)
        if not exists(matrix_path):
            raise FileNotFoundError(f"椭圆模式缺少 matrix_A.pt，请将其放在权重同目录下: {matrix_path}")

    return ModelAsset(mode=mode, weights_path=weights_path, matrix_path=matrix_path)
