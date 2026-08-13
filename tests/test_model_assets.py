import os

import pytest

from backend.model_assets import (
    MODE_CENTER,
    MODE_ELLIPSE,
    matrix_path_for_weights,
    validate_model_asset,
)


def test_matrix_path_for_weights_uses_weight_directory():
    assert matrix_path_for_weights(os.path.join("models", "ellipse.pth")) == os.path.join("models", "matrix_A.pt")


def test_validate_center_asset_requires_weight_file():
    with pytest.raises(FileNotFoundError, match="权重文件不存在"):
        validate_model_asset(MODE_CENTER, "missing.pth", exists=lambda _: False)


def test_validate_ellipse_asset_requires_matrix_file():
    def exists(path):
        return path.endswith("ellipse.pth")

    with pytest.raises(FileNotFoundError, match="matrix_A.pt"):
        validate_model_asset(MODE_ELLIPSE, os.path.join("models", "ellipse.pth"), exists=exists)


def test_validate_ellipse_asset_returns_matrix_path():
    asset = validate_model_asset(MODE_ELLIPSE, os.path.join("models", "ellipse.pth"), exists=lambda _: True)

    assert asset.mode == MODE_ELLIPSE
    assert asset.matrix_path == os.path.join("models", "matrix_A.pt")
