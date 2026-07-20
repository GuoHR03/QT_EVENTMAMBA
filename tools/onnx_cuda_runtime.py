"""Load NVIDIA wheel DLLs for ONNX Runtime on Windows."""

import os
import ctypes
from pathlib import Path

import onnxruntime as ort

_DLL_DIRECTORY_HANDLES = []
_DLL_HANDLES = []


def preload_cuda_dlls():
    site_packages = Path(ort.__file__).resolve().parent.parent
    candidates = (
        site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
        site_packages / "nvidia" / "cudnn" / "bin",
    )
    loaded_from = []
    for directory in candidates:
        if directory.is_dir():
            if os.name == "nt":
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
            loaded_from.append(str(directory))
    if not loaded_from:
        ort.preload_dlls(directory="")
    elif os.name == "nt":
        cuda_directory, cudnn_directory = candidates
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
    return loaded_from
