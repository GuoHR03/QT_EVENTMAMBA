# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import warnings
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


project_root = Path(SPECPATH)
venv_root = project_root / ".venv-onnx-win"
site_packages = venv_root / "Lib" / "site-packages"
python_prefix = Path(sys.base_prefix)
conda_bin = python_prefix / "Library" / "bin"

if not site_packages.is_dir():
    raise SystemExit(
        "Windows backend environment is missing: "
        f"{site_packages}. Create .venv-onnx-win before building."
    )

if Path(sys.prefix).resolve() != venv_root.resolve():
    raise SystemExit(
        "Build the backend with its dedicated interpreter: "
        r".venv-onnx-win\Scripts\python.exe -m PyInstaller "
        "UI_Event_Backend.spec"
    )

# Keep the venv ahead of any build-tool bootstrap path (for example, a
# temporary PYTHONPATH used only to expose PyInstaller itself).
sys.path.insert(0, str(site_packages))

for package_dir in ("onnxruntime", "numpy", "zmq"):
    if not (site_packages / package_dir).exists():
        raise SystemExit(
            f"Required backend package is missing from {site_packages}: {package_dir}"
        )


def collect_directory_dlls(source_dir, destination):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    return [
        (
            str(path),
            str(Path(destination) / path.relative_to(source_dir).parent),
        )
        for path in source_dir.rglob("*.dll")
    ]


def optional_conda_binary(name):
    path = conda_bin / name
    if path.is_file():
        return [(str(path), ".")]
    return []


def is_conflicting_libexpat(entry):
    source = Path(entry[1])
    return (
        source.name.lower() == "libexpat.dll"
        and source.parent.resolve() != conda_bin.resolve()
    )


def optional_cuda_12_runtime():
    candidates = []
    configured_root = os.environ.get("CUDA_PATH_V12_2")
    if configured_root:
        candidates.append(Path(configured_root) / "bin" / "cudart64_12.dll")
    candidates.append(
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2")
        / "bin"
        / "cudart64_12.dll"
    )
    for candidate in candidates:
        if candidate.is_file():
            return [(str(candidate), ".")]
    warnings.warn(
        "cudart64_12.dll was not found. The selective-scan custom operator "
        "requires this runtime; install CUDA 12.2 or set CUDA_PATH_V12_2 "
        "before building.",
        stacklevel=1,
    )
    return []


binaries = []
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("nvidia")
binaries += collect_directory_dlls(site_packages / "numpy.libs", "numpy.libs")
binaries += collect_directory_dlls(site_packages / "pyzmq.libs", ".")
binaries += sum(
    (
        optional_conda_binary(name)
        for name in (
            "ffi.dll",
            "libbz2.dll",
            "LIBBZ2.dll",
            "libcrypto-3-x64.dll",
            "libexpat.dll",
            "libmpdec-4.dll",
            "libssl-3-x64.dll",
            "sqlite3.dll",
        )
    ),
    [],
)
binaries += optional_cuda_12_runtime()

datas = collect_data_files(
    "onnxruntime",
    excludes=["**/__pycache__/**", "datasets/**"],
)
pyzmq_libs = site_packages / "pyzmq.libs"
if pyzmq_libs.is_dir():
    datas += [
        (str(path), "pyzmq.libs")
        for path in pyzmq_libs.glob(".load*")
        if path.is_file()
    ]

metadata_distributions = (
    "onnxruntime-gpu",
    "numpy",
    "pyzmq",
    "nvidia-cublas",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-nvjitlink",
)
for distribution in metadata_distributions:
    try:
        datas += copy_metadata(distribution)
    except Exception as exc:
        warnings.warn(
            f"Could not collect optional metadata for {distribution}: {exc}",
            stacklevel=1,
        )

hiddenimports = [
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "zmq.utils.garbage",
]
hiddenimports += collect_submodules("zmq.backend.cython")

a = Analysis(
    [str(project_root / "windows_backend.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "cv2",
        "h5py",
        "IPython",
        "matplotlib",
        "mamba_ssm",
        "pytest",
        "scipy",
        "sphinx",
        "timm",
        "tkinter",
        "torch",
        "yaml",
    ],
    noarchive=False,
    optimize=0,
)
a.binaries = [
    entry for entry in a.binaries if not is_conflicting_libexpat(entry)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UI_Event_Backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="UI_Event_Backend",
)
