# EventMamba Windows Selective Scan POC

This directory contains a Windows-native, inference-only CUDA custom operator
for ONNX Runtime. It replaces the six exported selective-scan `Loop` nodes that
made the original GPU graph much slower than CPU.

The implementation is deliberately small and is based on the selective state
recurrence already represented by this project's ONNX graph. It does not copy
the PyTorch extension or depend on `mamba-ssm` at runtime.

## Current constraints

- Windows x64 and NVIDIA CUDA only.
- FP32 inference only; no backward pass.
- State size must be at most 32 (the current model uses 16).
- The POC build targets CUDA architectures 7.5 and 8.6.
- ONNX Runtime 1.27 C API and `CUDAExecutionProvider` are required.

## Build and verify

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_selective_scan_ort.ps1
.venv-onnx-win/Scripts/python.exe tools/onnx_selective_scan_custom_op_probe.py
```

Rewrite and benchmark the complete center model:

```powershell
.venv-onnx-win/Scripts/python.exe tools/onnx_replace_selective_scan_loops.py `
  --input artifacts/eventmamba_center_windows.onnx `
  --output artifacts/eventmamba_center_selective_scan_cuda.onnx

.venv-onnx-win/Scripts/python.exe tools/onnx_windows_runtime_probe.py `
  --model artifacts/eventmamba_center_selective_scan_cuda.onnx `
  --provider CUDAExecutionProvider `
  --custom-op-library native/selective_scan_ort/bin/eventmamba_selective_scan.dll `
  --sample artifacts/real_raw_sample.npz `
  --repeats 10
```

The build script downloads only the two official ONNX Runtime 1.27 headers into
the ignored `.native-cache` directory. Intermediate build files remain ignored;
the verified runtime DLL is copied to `native/selective_scan_ort/bin/`.
