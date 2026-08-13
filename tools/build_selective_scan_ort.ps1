param(
    [string]$BuildDir = "native/selective_scan_ort/build/vs174_mismatch",
    [string]$OrtIncludeDir = ".native-cache/onnxruntime-1.27.0/include",
    [string]$CudaRoot = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $projectRoot "native/selective_scan_ort"
$resolvedBuildDir = Join-Path $projectRoot $BuildDir
$resolvedOrtInclude = Join-Path $projectRoot $OrtIncludeDir
$cmake = "E:/VS/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
$ninja = "E:/VS/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
$vsDevCmd = "E:/VS/Common7/Tools/VsDevCmd.bat"

foreach ($required in @($cmake, $ninja, $vsDevCmd)) {
    if (-not (Test-Path $required)) {
        throw "Required build input not found: $required"
    }
}

$ortHeaders = @{
    "onnxruntime_c_api.h" = "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.27.0/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "onnxruntime_ep_c_api.h" = "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.27.0/include/onnxruntime/core/session/onnxruntime_ep_c_api.h"
}
New-Item -ItemType Directory -Force -Path $resolvedOrtInclude | Out-Null
foreach ($entry in $ortHeaders.GetEnumerator()) {
    $destination = Join-Path $resolvedOrtInclude $entry.Key
    if (-not (Test-Path $destination)) {
        Write-Output "Downloading ONNX Runtime 1.27 header: $($entry.Key)"
        & curl.exe -L --fail --retry 5 -o $destination $entry.Value
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to download ONNX Runtime header: $($entry.Key)"
        }
    }
}

$environmentDump = & cmd.exe /d /s /c "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize the Visual Studio build environment"
}
foreach ($line in $environmentDump) {
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) {
        [Environment]::SetEnvironmentVariable(
            $line.Substring(0, $separator),
            $line.Substring($separator + 1),
            "Process"
        )
    }
}

$env:CUDAToolkit_ROOT = $CudaRoot
$cudaCompatibilityFlags = "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
$env:NVCC_PREPEND_FLAGS = $cudaCompatibilityFlags
& $cmake `
    -S $sourceDir `
    -B $resolvedBuildDir `
    -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_CUDA_FLAGS=$cudaCompatibilityFlags" `
    "-DCUDAToolkit_ROOT=$CudaRoot" `
    "-DONNXRUNTIME_INCLUDE_DIR=$resolvedOrtInclude"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed"
}

& $cmake --build $resolvedBuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "CUDA custom operator build failed"
}

$dll = Join-Path $resolvedBuildDir "eventmamba_selective_scan.dll"
if (-not (Test-Path $dll)) {
    throw "Build completed but DLL was not found: $dll"
}
$runtimeBin = Join-Path $sourceDir "bin"
New-Item -ItemType Directory -Force -Path $runtimeBin | Out-Null
$runtimeDll = Join-Path $runtimeBin "eventmamba_selective_scan.dll"
Copy-Item -LiteralPath $dll -Destination $runtimeDll -Force
Write-Output $runtimeDll
