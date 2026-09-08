param(
    [string]$BuildDir = "native/selective_scan_ort/build/release",
    [string]$OrtIncludeDir = ".native-cache/onnxruntime-1.27.0/include",
    [string]$CudaRoot,
    [string]$CMakePath,
    [string]$NinjaPath,
    [string]$VsDevCmdPath,
    [string]$VsInstallPath,
    [string]$VsWherePath,
    [switch]$ResolveOnly,
    [switch]$NoDeploy
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $ProjectRoot "native/selective_scan_ort"

function Resolve-ProjectPath {
    param([string]$Path)

    $expanded = [Environment]::ExpandEnvironmentVariables($Path.Trim().Trim('"'))
    if ([System.IO.Path]::IsPathRooted($expanded)) {
        return [System.IO.Path]::GetFullPath($expanded)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $expanded))
}

function Resolve-FileInput {
    param(
        [string]$Value,
        [string]$Label
    )

    if (-not $Value) {
        return $null
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Value.Trim().Trim('"'))
    $candidates = @()
    if ([System.IO.Path]::IsPathRooted($expanded)) {
        $candidates += $expanded
    } else {
        $candidates += (Join-Path $ProjectRoot $expanded)
        $command = Get-Command $expanded -CommandType Application -ErrorAction SilentlyContinue
        if ($command) {
            $candidates += $command.Source
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label was not found: $Value"
}

function Resolve-DirectoryInput {
    param(
        [string]$Value,
        [string]$Label
    )

    if (-not $Value) {
        return $null
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Value.Trim().Trim('"'))
    $candidate = if ([System.IO.Path]::IsPathRooted($expanded)) {
        $expanded
    } else {
        Join-Path $ProjectRoot $expanded
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "$Label was not found: $Value"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Find-VsWhere {
    if ($VsWherePath) {
        return Resolve-FileInput $VsWherePath "vswhere.exe"
    }
    if ($env:VSWHERE) {
        return Resolve-FileInput $env:VSWHERE "vswhere.exe from VSWHERE"
    }

    $command = Get-Command "vswhere.exe" -CommandType Application -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Find-VsInstallation {
    if ($VsInstallPath) {
        return Resolve-DirectoryInput $VsInstallPath "Visual Studio installation"
    }
    if ($env:VSINSTALLDIR -and (Test-Path -LiteralPath $env:VSINSTALLDIR -PathType Container)) {
        return (Resolve-Path -LiteralPath $env:VSINSTALLDIR).Path
    }

    $vswhere = Find-VsWhere
    if (-not $vswhere) {
        return $null
    }
    $installation = & $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installation) {
        return $null
    }
    return Resolve-DirectoryInput ($installation | Select-Object -First 1) "Visual Studio installation"
}

function Find-Tool {
    param(
        [string]$ExplicitPath,
        [string]$EnvironmentPath,
        [string]$CommandName,
        [string]$VisualStudioRelativePath,
        [string]$Label,
        [string]$VisualStudioRoot
    )

    if ($ExplicitPath) {
        return Resolve-FileInput $ExplicitPath $Label
    }
    if ($EnvironmentPath) {
        return Resolve-FileInput $EnvironmentPath "$Label from environment"
    }
    $command = Get-Command $CommandName -CommandType Application -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if ($VisualStudioRoot) {
        $candidate = Join-Path $VisualStudioRoot $VisualStudioRelativePath
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label was not found. Pass an explicit path or add it to PATH."
}

function Find-CudaRoot {
    if ($CudaRoot) {
        $candidates = @($CudaRoot)
    } else {
        $candidates = @(
            $env:CUDA_PATH_V12_2,
            $env:CUDA_PATH,
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
        )
    }

    foreach ($candidateValue in $candidates) {
        if (-not $candidateValue) {
            continue
        }
        try {
            $candidate = Resolve-DirectoryInput $candidateValue "CUDA Toolkit root"
        } catch {
            if ($CudaRoot) {
                throw
            }
            continue
        }
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        $runtimeHeader = Join-Path $candidate "include\cuda_runtime.h"
        if (
            (Test-Path -LiteralPath $nvcc -PathType Leaf) -and
            (Test-Path -LiteralPath $runtimeHeader -PathType Leaf)
        ) {
            return $candidate
        }
        if ($CudaRoot) {
            throw "CUDA Toolkit root is incomplete; expected bin\nvcc.exe and include\cuda_runtime.h: $candidate"
        }
    }
    throw "CUDA Toolkit 12.2 was not found. Pass -CudaRoot or set CUDA_PATH_V12_2."
}

function Save-RemoteFile {
    param(
        [string]$Uri,
        [string]$Destination
    )

    $partial = "$Destination.part"
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Uri -OutFile $partial -UseBasicParsing
            Move-Item -LiteralPath $partial -Destination $Destination -Force
            return
        } catch {
            if ($attempt -eq 5) {
                throw "Failed to download $Uri after $attempt attempts: $($_.Exception.Message)"
            }
            Start-Sleep -Seconds 2
        }
    }
}

$ResolvedBuildDir = Resolve-ProjectPath $BuildDir
$ResolvedOrtInclude = Resolve-ProjectPath $OrtIncludeDir
$ResolvedVsInstall = Find-VsInstallation
$ResolvedVsDevCmd = if ($VsDevCmdPath) {
    Resolve-FileInput $VsDevCmdPath "VsDevCmd.bat"
} elseif ($env:VSDEVCMD) {
    Resolve-FileInput $env:VSDEVCMD "VsDevCmd.bat from VSDEVCMD"
} elseif ($ResolvedVsInstall) {
    Resolve-FileInput (Join-Path $ResolvedVsInstall "Common7\Tools\VsDevCmd.bat") "VsDevCmd.bat"
} else {
    throw "Visual Studio C++ tools were not found. Pass -VsInstallPath or -VsDevCmdPath."
}
$ResolvedCMake = Find-Tool `
    $CMakePath `
    $env:CMAKE_EXE `
    "cmake.exe" `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
    "CMake" `
    $ResolvedVsInstall
$ResolvedNinja = Find-Tool `
    $NinjaPath `
    $env:NINJA_EXE `
    "ninja.exe" `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" `
    "Ninja" `
    $ResolvedVsInstall
$ResolvedCudaRoot = Find-CudaRoot

$ResolvedInputs = [ordered]@{
    project_root = $ProjectRoot
    source_dir = $SourceDir
    build_dir = $ResolvedBuildDir
    onnxruntime_include_dir = $ResolvedOrtInclude
    visual_studio = $ResolvedVsInstall
    vsdevcmd = $ResolvedVsDevCmd
    cmake = $ResolvedCMake
    ninja = $ResolvedNinja
    cuda_root = $ResolvedCudaRoot
}

if ($ResolveOnly) {
    $ResolvedInputs | ConvertTo-Json
    exit 0
}

Write-Host "==> Resolved native build inputs"
$ResolvedInputs.GetEnumerator() | ForEach-Object {
    Write-Host "    $($_.Key): $($_.Value)"
}

$OrtHeaders = @{
    "onnxruntime_c_api.h" = "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.0/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "onnxruntime_ep_c_api.h" = "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.0/include/onnxruntime/core/session/onnxruntime_ep_c_api.h"
}
New-Item -ItemType Directory -Force -Path $ResolvedOrtInclude | Out-Null
foreach ($entry in $OrtHeaders.GetEnumerator()) {
    $destination = Join-Path $ResolvedOrtInclude $entry.Key
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        Write-Host "==> Downloading ONNX Runtime 1.27 header: $($entry.Key)"
        Save-RemoteFile $entry.Value $destination
    }
    if ((Get-Item -LiteralPath $destination).Length -le 0) {
        throw "ONNX Runtime header is empty: $destination"
    }
}

$environmentDump = & cmd.exe /d /s /c "`"$ResolvedVsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
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

$env:CUDAToolkit_ROOT = $ResolvedCudaRoot
$cudaCompatibilityFlags = "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
$env:NVCC_PREPEND_FLAGS = $cudaCompatibilityFlags
& $ResolvedCMake `
    -S $SourceDir `
    -B $ResolvedBuildDir `
    -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ResolvedNinja" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_CUDA_FLAGS=$cudaCompatibilityFlags" `
    "-DCUDAToolkit_ROOT=$ResolvedCudaRoot" `
    "-DONNXRUNTIME_INCLUDE_DIR=$ResolvedOrtInclude"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed"
}

& $ResolvedCMake --build $ResolvedBuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "CUDA custom operator build failed"
}

$dll = Join-Path $ResolvedBuildDir "eventmamba_selective_scan.dll"
if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) {
    throw "Build completed but DLL was not found: $dll"
}
if ($NoDeploy) {
    Write-Host "==> Native custom operator verification build ready: $dll"
    exit 0
}
$runtimeBin = Join-Path $SourceDir "bin"
New-Item -ItemType Directory -Force -Path $runtimeBin | Out-Null
$runtimeDll = Join-Path $runtimeBin "eventmamba_selective_scan.dll"
Copy-Item -LiteralPath $dll -Destination $runtimeDll -Force
Write-Host "==> Native custom operator ready: $runtimeDll"
