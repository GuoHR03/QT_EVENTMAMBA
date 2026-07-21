from PyQt6.QtWidgets import QFileDialog

try:
    from .paths import default_checkpoint_dir, default_onnx_model_dir, default_record_dir
except ImportError:
    from paths import default_checkpoint_dir, default_onnx_model_dir, default_record_dir


def choose_input_file(parent):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择离线视频文件",
        default_record_dir(),
        "视频文件 (*.raw *.hdf5 *.h5 *.aedat4);;所有文件 (*)",
    )
    return file_path


def choose_weights_file(parent, runtime_kind="wsl"):
    if runtime_kind == "windows":
        title = "选择 Windows ONNX 模型"
        initial_dir = default_onnx_model_dir()
        file_filter = "ONNX 模型 (*.onnx);;所有文件 (*)"
    else:
        title = "选择 PyTorch 权重"
        initial_dir = default_checkpoint_dir()
        file_filter = "PyTorch 权重 (*.pth);;所有文件 (*)"

    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        title,
        initial_dir,
        file_filter,
    )
    return file_path
