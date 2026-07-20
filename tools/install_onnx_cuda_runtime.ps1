param(
    [string]$PythonPath = ".\.venv-onnx-win\Scripts\python.exe",
    [string]$CacheDirectory = ".\.onnx-wheel-cache"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found: $PythonPath"
}

New-Item -ItemType Directory -Force -Path $CacheDirectory | Out-Null

# Versions match onnxruntime-gpu 1.27.0 (CUDA 13.x / cuDNN 9.x).
$packages = @(
    @{
        File = "nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-cuda-runtime/nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_cuda_nvrtc-13.0.88-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-cuda-nvrtc/nvidia_cuda_nvrtc-13.0.88-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_nvjitlink-13.0.88-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-nvjitlink/nvidia_nvjitlink-13.0.88-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-cufft/nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_curand-10.4.0.35-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-curand/nvidia_curand-10.4.0.35-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_cublas-13.0.2.14-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-cublas/nvidia_cublas-13.0.2.14-py3-none-win_amd64.whl"
    },
    @{
        File = "nvidia_cudnn_cu13-9.13.1.26-py3-none-win_amd64.whl"
        Url = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.13.1.26-py3-none-win_amd64.whl"
    }
)

foreach ($package in $packages) {
    $destination = Join-Path $CacheDirectory $package.File
    $partial = "$destination.part"
    if (Test-Path -LiteralPath $destination) {
        Write-Host "[cached] $($package.File)"
        continue
    }

    Write-Host "[download] $($package.File)"
    & curl.exe `
        --fail `
        --location `
        --continue-at - `
        --retry 20 `
        --retry-all-errors `
        --retry-delay 5 `
        --connect-timeout 30 `
        --speed-limit 1024 `
        --speed-time 120 `
        --output $partial `
        $package.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code $LASTEXITCODE. Re-run this script to resume: $($package.File)"
    }
    Move-Item -LiteralPath $partial -Destination $destination -Force
}

$wheelPaths = $packages | ForEach-Object {
    (Resolve-Path -LiteralPath (Join-Path $CacheDirectory $_.File)).Path
}

Write-Host "[install] Installing cached NVIDIA runtime wheels"
& $PythonPath -m pip install --no-index --no-deps @wheelPaths
if ($LASTEXITCODE -ne 0) {
    throw "Offline wheel installation failed with exit code $LASTEXITCODE"
}

Write-Host "[verify] Preloading CUDA and cuDNN DLLs"
& $PythonPath -c "import onnxruntime as ort; from tools.onnx_cuda_runtime import preload_cuda_dlls; print('loaded_from=', preload_cuda_dlls()); print('providers=', ort.get_available_providers())"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DLL preload verification failed with exit code $LASTEXITCODE"
}

Write-Host "[done] NVIDIA CUDA runtime installation completed"
