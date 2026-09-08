import json
import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "tools" / "build_selective_scan_ort.ps1"


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="The native operator build resolver targets Windows tooling",
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _run_resolver(*arguments):
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            "-ResolveOnly",
            *map(str, arguments),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_native_build_resolver_honors_explicit_portable_paths(tmp_path):
    vs_root = tmp_path / "Visual Studio"
    vsdevcmd = _touch(vs_root / "Common7" / "Tools" / "VsDevCmd.bat")
    cmake = _touch(tmp_path / "tools" / "cmake.exe")
    ninja = _touch(tmp_path / "tools" / "ninja.exe")
    cuda_root = tmp_path / "CUDA" / "v12.2"
    _touch(cuda_root / "bin" / "nvcc.exe")
    _touch(cuda_root / "include" / "cuda_runtime.h")

    result = _run_resolver(
        "-VsInstallPath",
        vs_root,
        "-CMakePath",
        cmake,
        "-NinjaPath",
        ninja,
        "-CudaRoot",
        cuda_root,
    )

    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert Path(resolved["visual_studio"]) == vs_root
    assert Path(resolved["vsdevcmd"]) == vsdevcmd
    assert Path(resolved["cmake"]) == cmake
    assert Path(resolved["ninja"]) == ninja
    assert Path(resolved["cuda_root"]) == cuda_root


def test_native_build_resolver_rejects_incomplete_explicit_cuda_root(tmp_path):
    vs_root = tmp_path / "vs"
    _touch(vs_root / "Common7" / "Tools" / "VsDevCmd.bat")
    cmake = _touch(tmp_path / "cmake.exe")
    ninja = _touch(tmp_path / "ninja.exe")
    cuda_root = tmp_path / "incomplete-cuda"
    cuda_root.mkdir()

    result = _run_resolver(
        "-VsInstallPath",
        vs_root,
        "-CMakePath",
        cmake,
        "-NinjaPath",
        ninja,
        "-CudaRoot",
        cuda_root,
    )

    assert result.returncode != 0
    assert "expected bin\\nvcc.exe and include\\cuda_runtime.h" in result.stderr
