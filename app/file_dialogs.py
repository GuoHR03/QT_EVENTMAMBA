from PyQt6.QtWidgets import QFileDialog

try:
    from .paths import default_checkpoint_dir, default_record_dir
except ImportError:
    from paths import default_checkpoint_dir, default_record_dir


def choose_input_file(parent):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择离线视频文件",
        default_record_dir(),
        "视频文件 (*.raw *.hdf5 *.h5 *.aedat4);;所有文件 (*)",
    )
    return file_path


def choose_weights_file(parent):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择权重文件",
        default_checkpoint_dir(),
        "权重文件 (*.pth);;所有文件 (*)",
    )
    return file_path
