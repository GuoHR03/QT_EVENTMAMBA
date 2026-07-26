import sys
from types import SimpleNamespace

from backend import windows_onnx_runtime


def test_prepare_windows_cuda_runtime_preloads_bundled_cuda12(tmp_path, monkeypatch):
    site_packages = tmp_path / "_internal"
    ort_package = site_packages / "onnxruntime"
    cuda_directory = site_packages / "nvidia" / "cu13" / "bin" / "x86_64"
    cudnn_directory = site_packages / "nvidia" / "cudnn" / "bin"
    system_cuda_directory = tmp_path / "cuda-12.2" / "bin"
    for directory in (
        ort_package,
        cuda_directory,
        cudnn_directory,
        system_cuda_directory,
    ):
        directory.mkdir(parents=True)

    bundled_cuda_runtime = site_packages / "cudart64_12.dll"
    bundled_cuda_runtime.touch()
    fake_ort = SimpleNamespace(
        __file__=str(ort_package / "__init__.py"),
        preload_dlls=lambda directory="": None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setenv("CUDA_PATH_V12_2", str(tmp_path / "cuda-12.2"))

    registered_directories = []
    loaded_libraries = []
    monkeypatch.setattr(
        windows_onnx_runtime,
        "_add_dll_directory",
        lambda directory: registered_directories.append(directory) or True,
    )
    monkeypatch.setattr(
        windows_onnx_runtime.ctypes,
        "WinDLL",
        lambda path: loaded_libraries.append(path) or object(),
    )
    windows_onnx_runtime._DLL_HANDLES.clear()

    loaded_directories = windows_onnx_runtime.prepare_windows_cuda_runtime()

    assert registered_directories == [
        site_packages,
        cuda_directory,
        cudnn_directory,
        system_cuda_directory,
    ]
    assert loaded_directories == [str(path) for path in registered_directories]
    assert loaded_libraries == [str(bundled_cuda_runtime)]
