"""Windows DLL preparation for the ONNX Runtime CUDA execution provider."""

import ctypes
import os
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_DLL_HANDLES = []


def _add_dll_directory(directory):
    if os.name == "nt" and directory.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        return True
    return False


def prepare_windows_cuda_runtime():
    import onnxruntime as ort

    site_packages = Path(ort.__file__).resolve().parent.parent
    cuda_directory = site_packages / "nvidia" / "cu13" / "bin" / "x86_64"
    cudnn_directory = site_packages / "nvidia" / "cudnn" / "bin"
    cuda_12_directory = Path(
        os.environ.get(
            "CUDA_PATH_V12_2",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        )
    ) / "bin"

    loaded_directories = []
    for directory in (
        site_packages,
        cuda_directory,
        cudnn_directory,
        cuda_12_directory,
    ):
        if _add_dll_directory(directory):
            loaded_directories.append(str(directory))

    bundled_cuda_12_runtime = site_packages / "cudart64_12.dll"
    if bundled_cuda_12_runtime.is_file():
        _DLL_HANDLES.append(ctypes.WinDLL(str(bundled_cuda_12_runtime)))

    if not cuda_directory.is_dir() or not cudnn_directory.is_dir():
        ort.preload_dlls(directory="")
        return loaded_directories

    for name in (
        "cudart64_13.dll",
        "cublasLt64_13.dll",
        "cublas64_13.dll",
        "cufft64_12.dll",
        "curand64_10.dll",
        "nvJitLink_130_0.dll",
        "nvrtc-builtins64_130.dll",
        "nvrtc64_130_0.dll",
    ):
        path = cuda_directory / name
        if path.is_file():
            _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
    for name in (
        "cudnn64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
    ):
        path = cudnn_directory / name
        if path.is_file():
            _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
    return loaded_directories
