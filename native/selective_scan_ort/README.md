# EventMamba Windows ONNX Runtime Custom Operators

This directory contains the Windows-native ONNX Runtime custom operators used
by EventMamba:

- CUDA `SelectiveScan` and `SelectiveScanCore` operators replace the exported
  selective-scan `Loop` nodes.
- CPU `HierarchicalFarthestPointSampling` moves the three-stage FPS pipeline
  out of Python while preserving the model's sampling contract.

The implementation is deliberately small and is based on the selective state
recurrence already represented by this project's ONNX graph. It does not copy
the PyTorch extension or depend on `mamba-ssm` at runtime.

## Current constraints

- Windows x64 and NVIDIA CUDA only.
- FP32 inference only; no backward pass.
- State size must be at most 32 (the current model uses 16).
- Hierarchical FPS accepts finite FP32 `[B,3,1024]` events and INT64 `[B,3]`
  start indices, and returns INT64 samples with sizes 512, 256 and 128.
- The POC build targets CUDA architectures 7.5 and 8.6.
- ONNX Runtime 1.27 C API and `CUDAExecutionProvider` are required.

## Build and verify

### Prerequisites

- Windows 10/11 x64.
- Visual Studio 2022 with the C++ desktop toolchain.
- CMake and Ninja.
- CUDA Toolkit 12.2.
- The project's `.venv-onnx-win` environment with ONNX Runtime GPU, NumPy and
  the probe dependencies.
- Network access on the first build so the script can download the two official
  ONNX Runtime 1.27 C API headers.

Important: `tools/build_selective_scan_ort.ps1` currently contains local
Visual Studio paths under `E:/VS/...`. A clean clone on another machine will
not build until those paths are changed or parameterized. The commands below
describe the verified project workflow, not yet a portable one-command SDK
bootstrap.

The release packaging entry point and final asset checks are documented in
[`PACKAGING.md`](../../PACKAGING.md). This file focuses only on custom-operator
development.

### Build the DLL and run standalone probes

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_selective_scan_ort.ps1
.venv-onnx-win/Scripts/python.exe tools/onnx_selective_scan_custom_op_probe.py
.venv-onnx-win/Scripts/python.exe tools/onnx_hierarchical_fps_custom_op_probe.py
```

The standalone FPS probe must match the NumPy oracle index-for-index.

### Rewrite and benchmark a complete model

The repository tracks `eventmamba_center_selective_scan_cuda.onnx`, which can
be used directly as the rewrite input:

```powershell
.venv-onnx-win/Scripts/python.exe tools/onnx_insert_hierarchical_fps.py `
  --input artifacts/eventmamba_center_selective_scan_cuda.onnx `
  --output artifacts/eventmamba_center_native_fps.onnx `
  --overwrite
```

The benchmark sample `artifacts/real_raw_sample.npz` is a local artifact and is
not tracked. Generate it from a RAW file before running the benchmark:

```powershell
.venv-onnx-win/Scripts/python.exe tools/extract_raw_inference_sample.py `
  record/your_sample.raw `
  artifacts/real_raw_sample.npz

.venv-onnx-win/Scripts/python.exe tools/onnx_windows_runtime_probe.py `
  --model artifacts/eventmamba_center_native_fps.onnx `
  --provider CUDAExecutionProvider `
  --custom-op-library native/selective_scan_ort/bin/eventmamba_selective_scan.dll `
  --sample artifacts/real_raw_sample.npz `
  --warmups 20 `
  --repeats 100
```

To regenerate the tracked `*_selective_scan_cuda.onnx` source model itself,
first export the unfused model and run `tools/onnx_replace_selective_scan_loops.py`.
Those export inputs depend on the original PyTorch checkpoint and are outside
the clean runtime-asset workflow.

The full model probe reports input preparation, ORT session, and end-to-end
P50/P95 separately so Python farthest-point-sampling time cannot be
accidentally omitted from the baseline.

The build script downloads only the two official ONNX Runtime 1.27 headers into
the ignored `.native-cache` directory. Intermediate build files remain ignored;
the verified runtime DLL is copied to `native/selective_scan_ort/bin/`.
