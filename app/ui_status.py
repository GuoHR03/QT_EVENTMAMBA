import os


SOURCE_DISPLAY_NAMES = {
    ".raw": "RAW 文件",
    ".h5": "H5 文件",
    ".hdf5": "H5 文件",
    ".aedat4": "AEDAT4 文件",
}


def source_display_name(file_path):
    if not file_path:
        return "实时输入"
    extension = os.path.splitext(str(file_path))[1].lower()
    return SOURCE_DISPLAY_NAMES.get(extension, "离线文件")
