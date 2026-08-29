param(
    [switch]$Clean,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot

$ReleaseVersion = "0.2.0"
$UiPython = Join-Path $ProjectRoot ".qtcreator\Pythonvenv\Scripts\python.exe"
$BackendPython = Join-Path $ProjectRoot ".venv-onnx-win\Scripts\python.exe"
$UiSpec = Join-Path $ProjectRoot "UI_Event.spec"
$BackendSpec = Join-Path $ProjectRoot "UI_Event_Backend.spec"
$InstallerScript = Join-Path $ProjectRoot "installer.iss"
$BuildRoot = Join-Path $ProjectRoot "build"
$DistRoot = Join-Path $ProjectRoot "dist"
$UiBundleDir = Join-Path $DistRoot "UI_Event"
$BackendBundleDir = Join-Path $DistRoot "UI_Event_Backend"
$BackendRuntimeDir = Join-Path $UiBundleDir "backend_runtime"
$ArtifactSourceDir = Join-Path $ProjectRoot "artifacts"
$NativeDllSource = Join-Path $ProjectRoot "native\selective_scan_ort\bin\eventmamba_selective_scan.dll"
$ArtifactValidator = Join-Path $ProjectRoot "tools\validate_windows_inference_artifacts.py"
$NativeFpsProbe = Join-Path $ProjectRoot "tools\onnx_hierarchical_fps_custom_op_probe.py"
$ProjectMetavisionRuntime = Join-Path $ProjectRoot "libs"
$MetavisionSdkRoot = if ($env:METAVISION_SDK_PATH) {
    [Environment]::ExpandEnvironmentVariables($env:METAVISION_SDK_PATH.Trim().Trim('"'))
} elseif (
    (Test-Path -LiteralPath (Join-Path $ProjectMetavisionRuntime "third_party\bin") -PathType Container) -and
    (Test-Path -LiteralPath (Join-Path $ProjectMetavisionRuntime "lib\hdf5\plugin") -PathType Container) -and
    (Test-Path -LiteralPath (Join-Path $ProjectMetavisionRuntime "lib\metavision\hal\plugins") -PathType Container)
) {
    $ProjectMetavisionRuntime
} else {
    "E:\Metavision\Prophesee"
}
$MetavisionDestinationRoot = Join-Path $UiBundleDir "metavision"
$MetavisionRuntimeFiles = @(
    "third_party\bin\boost_filesystem-vc143-mt-x64-1_78.dll",
    "third_party\bin\jpeg62.dll",
    "third_party\bin\liblzma.dll",
    "third_party\bin\libpng16.dll",
    "third_party\bin\libsharpyuv.dll",
    "third_party\bin\libusb-1.0.dll",
    "third_party\bin\libwebp.dll",
    "third_party\bin\libwebpdecoder.dll",
    "third_party\bin\opencv_core4.dll",
    "third_party\bin\opencv_highgui4.dll",
    "third_party\bin\opencv_imgcodecs4.dll",
    "third_party\bin\opencv_imgproc4.dll",
    "third_party\bin\opencv_videoio4.dll",
    "third_party\bin\tiff.dll",
    "third_party\bin\zlib1.dll",
    "lib\hdf5\plugin\H5Zecf.dll",
    "lib\metavision\hal\plugins\hal_plugin_prophesee.dll",
    "lib\metavision\hal\plugins\metavision_psee_hw_layer.dll"
)
$ArtifactNames = @(
    "eventmamba_center_native_fps.onnx",
    "eventmamba_ellipse_native_fps.onnx",
    "eventmamba_ellipse_matrix_A.npy"
)

function Assert-FileExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }

    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "$Label is empty: $Path"
    }
}

function Assert-DirectoryExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label was not found: $Path"
    }
}

function Assert-NoForbiddenReleaseFiles {
    param([string]$Root)

    $forbidden = Get-ChildItem -LiteralPath $Root -File -Recurse | Where-Object {
        $_.FullName -match '[\\/]__pycache__[\\/]' -or
        $_.Name -match '_d(?:\.dll|\.cp\d+-win_amd64\.pyd)$' -or
        $_.Name -match '\.cp39-win_amd64\.pyd$'
    }
    if ($forbidden) {
        $sample = ($forbidden | Select-Object -First 5 -ExpandProperty FullName) -join ', '
        throw "Forbidden debug/cache/CPython 3.9 files entered the release bundle: $sample"
    }
}

function Remove-GeneratedPath {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolvedTarget = (Resolve-Path -LiteralPath $Path).Path.TrimEnd("\")
    $allowedTargets = @(
        (Join-Path $ProjectRoot "build"),
        (Join-Path $ProjectRoot "dist"),
        (Join-Path $ProjectRoot "installer")
    ) | ForEach-Object {
        [System.IO.Path]::GetFullPath($_).TrimEnd("\")
    }

    if ($allowedTargets -notcontains $resolvedTarget) {
        throw "Refusing to remove anything except generated build, dist, or installer directories: $resolvedTarget"
    }

    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
}

function Assert-PyInstaller {
    param(
        [string]$Python,
        [string]$EnvironmentName
    )

    Write-Host "==> Checking PyInstaller in $EnvironmentName"
    & $Python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is not installed in $EnvironmentName. Install it with: `"$Python`" -m pip install pyinstaller"
    }
}

function Assert-InferenceArtifacts {
    param(
        [string]$CenterModel,
        [string]$EllipseModel,
        [string]$EllipseMatrix,
        [string]$CustomOpLibrary,
        [switch]$ProbeNativeFps
    )

    Write-Host "==> Validating native-FPS inference artifacts"
    & $BackendPython $ArtifactValidator `
        --center $CenterModel `
        --ellipse $EllipseModel `
        --matrix $EllipseMatrix `
        --custom-op-library $CustomOpLibrary
    if ($LASTEXITCODE -ne 0) {
        throw "Inference artifact contract validation failed"
    }

    if ($ProbeNativeFps) {
        & $BackendPython $NativeFpsProbe `
            --custom-op-library $CustomOpLibrary `
            --warmups 1 `
            --repeats 3
        if ($LASTEXITCODE -ne 0) {
            throw "Native hierarchical FPS custom-op probe failed"
        }
    }
}

function Invoke-PackagedBackendSmoke {
    param(
        [string]$BackendExecutable,
        [string]$CenterModel,
        [string]$EllipseModel,
        [string]$EllipseMatrix,
        [string]$CustomOpLibrary
    )

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        0
    )
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()

    $stdoutLog = Join-Path $BuildRoot "packaged_backend_smoke.stdout.log"
    $stderrLog = Join-Path $BuildRoot "packaged_backend_smoke.stderr.log"
    $arguments = @(
        "--center-model", "`"$CenterModel`"",
        "--ellipse-model", "`"$EllipseModel`"",
        "--ellipse-matrix", "`"$EllipseMatrix`"",
        "--custom-op-library", "`"$CustomOpLibrary`"",
        "--port", "$port",
        "--instance-nonce", "packaged-fps-smoke"
    )

    Write-Host "==> Running packaged backend native-FPS smoke on port $port"
    $process = Start-Process `
        -FilePath $BackendExecutable `
        -ArgumentList $arguments `
        -WorkingDirectory (Split-Path -Parent $BackendExecutable) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
    try {
        & $BackendPython $NativeFpsProbe `
            --custom-op-library $CustomOpLibrary `
            --warmups 1 `
            --repeats 3
        if ($LASTEXITCODE -ne 0) {
            throw "Staged native FPS operator probe failed"
        }
        & $BackendPython (Join-Path $ProjectRoot "tools\smoke_packaged_backend.py") `
            --port $port `
            --timeout-s 90
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged backend prediction smoke failed"
        }
    }
    catch {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
        }
        if (Test-Path -LiteralPath $stdoutLog) {
            Write-Host (Get-Content -LiteralPath $stdoutLog -Raw -ErrorAction SilentlyContinue)
        }
        if (Test-Path -LiteralPath $stderrLog) {
            Write-Host (Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue)
        }
        throw
    }
    finally {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
        }
    }
}

function Invoke-PyInstallerBuild {
    param(
        [string]$Python,
        [string]$Spec,
        [string]$WorkPath,
        [string]$Label
    )

    Write-Host "==> Building $Label"
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistRoot `
        --workpath $WorkPath `
        $Spec
    if ($LASTEXITCODE -ne 0) {
        throw "$Label PyInstaller build failed"
    }
}

function Find-InnoCompiler {
    if ($env:ISCC -and (Test-Path -LiteralPath $env:ISCC)) {
        return $env:ISCC
    }

    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

Assert-FileExists $UiPython "UI Python interpreter"
Assert-FileExists $BackendPython "Windows inference Python interpreter"
Assert-FileExists $UiSpec "UI PyInstaller spec"
Assert-FileExists $BackendSpec "Windows backend PyInstaller spec"
Assert-FileExists $InstallerScript "Inno Setup script"
Assert-FileExists $ArtifactValidator "Inference artifact validator"
Assert-FileExists $NativeFpsProbe "Native FPS probe"
$expectedInstallerVersion = "#define MyAppVersion `"$ReleaseVersion`""
if (-not (Select-String `
    -LiteralPath $InstallerScript `
    -SimpleMatch $expectedInstallerVersion `
    -Quiet)) {
    throw "Release version mismatch: installer.iss must contain $expectedInstallerVersion"
}
foreach ($artifactName in $ArtifactNames) {
    Assert-FileExists (Join-Path $ArtifactSourceDir $artifactName) "Inference artifact"
}
Assert-FileExists $NativeDllSource "Selective-scan custom operator"
Assert-DirectoryExists $MetavisionSdkRoot "Metavision SDK root (set METAVISION_SDK_PATH if installed elsewhere)"
foreach ($runtimeFile in $MetavisionRuntimeFiles) {
    Assert-FileExists `
        (Join-Path $MetavisionSdkRoot $runtimeFile) `
        "Required Metavision runtime file"
}

Write-Host "==> UI_Event release $ReleaseVersion"
Write-Host "==> Project: $ProjectRoot"
Write-Host "==> UI Python: $UiPython"
Write-Host "==> Backend Python: $BackendPython"
Write-Host "==> Metavision SDK: $MetavisionSdkRoot"

Assert-PyInstaller $UiPython "UI environment"
Assert-PyInstaller $BackendPython "Windows inference environment"

Assert-InferenceArtifacts `
    -CenterModel (Join-Path $ArtifactSourceDir "eventmamba_center_native_fps.onnx") `
    -EllipseModel (Join-Path $ArtifactSourceDir "eventmamba_ellipse_native_fps.onnx") `
    -EllipseMatrix (Join-Path $ArtifactSourceDir "eventmamba_ellipse_matrix_A.npy") `
    -CustomOpLibrary $NativeDllSource `
    -ProbeNativeFps

if ($Clean) {
    Write-Host "==> Cleaning generated build outputs"
    Remove-GeneratedPath $BuildRoot
    Remove-GeneratedPath $DistRoot
    Remove-GeneratedPath (Join-Path $ProjectRoot "installer")
}

Invoke-PyInstallerBuild `
    -Python $BackendPython `
    -Spec $BackendSpec `
    -WorkPath (Join-Path $BuildRoot "UI_Event_Backend") `
    -Label "UI_Event_Backend"

$backendExePath = Join-Path $BackendBundleDir "UI_Event_Backend.exe"
Assert-FileExists $backendExePath "Windows backend executable"
Assert-DirectoryExists (Join-Path $BackendBundleDir "_internal") "Windows backend runtime directory"

Invoke-PyInstallerBuild `
    -Python $UiPython `
    -Spec $UiSpec `
    -WorkPath (Join-Path $BuildRoot "UI_Event") `
    -Label "UI_Event"

$uiExePath = Join-Path $UiBundleDir "UI_Event.exe"
Assert-FileExists $uiExePath "UI executable"
Assert-DirectoryExists (Join-Path $UiBundleDir "_internal") "UI runtime directory"

if (Test-Path -LiteralPath $BackendRuntimeDir) {
    throw "Reserved staging directory already exists in the fresh UI bundle: $BackendRuntimeDir"
}

Write-Host "==> Staging Windows inference runtime"
$resolvedBackendBundle = [System.IO.Path]::GetFullPath($BackendBundleDir).TrimEnd("\")
$expectedBackendBundle = [System.IO.Path]::GetFullPath(
    (Join-Path $DistRoot "UI_Event_Backend")
).TrimEnd("\")
if ($resolvedBackendBundle -ne $expectedBackendBundle) {
    throw "Refusing to move an unexpected backend bundle: $resolvedBackendBundle"
}
Move-Item -LiteralPath $resolvedBackendBundle -Destination $BackendRuntimeDir
if (Test-Path -LiteralPath $BackendBundleDir) {
    throw "Backend staging left a duplicate bundle behind: $BackendBundleDir"
}

$artifactDestinationDir = Join-Path $UiBundleDir "artifacts"
New-Item -ItemType Directory -Path $artifactDestinationDir -Force | Out-Null
foreach ($artifactName in $ArtifactNames) {
    Copy-Item `
        -LiteralPath (Join-Path $ArtifactSourceDir $artifactName) `
        -Destination (Join-Path $artifactDestinationDir $artifactName) `
        -Force
}

$nativeDllDestination = Join-Path $UiBundleDir "native\selective_scan_ort\bin\eventmamba_selective_scan.dll"
$nativeDllDestinationDir = Split-Path -Parent $nativeDllDestination
New-Item -ItemType Directory -Path $nativeDllDestinationDir -Force | Out-Null
Copy-Item -LiteralPath $NativeDllSource -Destination $nativeDllDestination -Force

Write-Host "==> Staging Metavision runtime"
foreach ($runtimeFile in $MetavisionRuntimeFiles) {
    $runtimeSource = Join-Path $MetavisionSdkRoot $runtimeFile
    $runtimeDestination = Join-Path $MetavisionDestinationRoot $runtimeFile
    New-Item `
        -ItemType Directory `
        -Path (Split-Path -Parent $runtimeDestination) `
        -Force |
        Out-Null
    Copy-Item `
        -LiteralPath $runtimeSource `
        -Destination $runtimeDestination `
        -Force
}

$requiredBundleFiles = @(
    $uiExePath,
    (Join-Path $BackendRuntimeDir "UI_Event_Backend.exe"),
    (Join-Path $artifactDestinationDir "eventmamba_center_native_fps.onnx"),
    (Join-Path $artifactDestinationDir "eventmamba_ellipse_native_fps.onnx"),
    (Join-Path $artifactDestinationDir "eventmamba_ellipse_matrix_A.npy"),
    $nativeDllDestination,
    (Join-Path $UiBundleDir "_internal\app\form.ui"),
    (Join-Path $UiBundleDir "_internal\metavision_hal_internal.cp38-win_amd64.pyd"),
    (Join-Path $UiBundleDir "_internal\metavision_sdk_base_internal.cp38-win_amd64.pyd"),
    (Join-Path $UiBundleDir "_internal\metavision_sdk_base_paths_internal.cp38-win_amd64.pyd"),
    (Join-Path $UiBundleDir "_internal\metavision_sdk_core_internal.cp38-win_amd64.pyd"),
    (Join-Path $UiBundleDir "_internal\metavision_sdk_cv_internal.cp38-win_amd64.pyd")
)
foreach ($requiredFile in $requiredBundleFiles) {
    Assert-FileExists $requiredFile "Required release file"
}
Assert-InferenceArtifacts `
    -CenterModel (Join-Path $artifactDestinationDir "eventmamba_center_native_fps.onnx") `
    -EllipseModel (Join-Path $artifactDestinationDir "eventmamba_ellipse_native_fps.onnx") `
    -EllipseMatrix (Join-Path $artifactDestinationDir "eventmamba_ellipse_matrix_A.npy") `
    -CustomOpLibrary $nativeDllDestination
Assert-DirectoryExists (Join-Path $UiBundleDir "_internal\libs\bin") "Bundled Metavision release DLLs"
Assert-DirectoryExists (Join-Path $BackendRuntimeDir "_internal") "Bundled Windows inference dependencies"
foreach ($runtimeFile in $MetavisionRuntimeFiles) {
    Assert-FileExists `
        (Join-Path $MetavisionDestinationRoot $runtimeFile) `
        "Staged Metavision runtime file"
}

$unexpectedSourceCopies = @(
    (Join-Path $UiBundleDir "_internal\backend"),
    (Join-Path $UiBundleDir "_internal\linux_backend.py"),
    (Join-Path $UiBundleDir "_internal\windows_backend.py")
)
foreach ($unexpectedPath in $unexpectedSourceCopies) {
    if (Test-Path -LiteralPath $unexpectedPath) {
        throw "Source directory/file was copied into the frozen UI bundle: $unexpectedPath"
    }
}
Assert-NoForbiddenReleaseFiles $UiBundleDir

$cuda12Runtime = Get-ChildItem `
    -LiteralPath (Join-Path $BackendRuntimeDir "_internal") `
    -Filter "cudart64_12.dll" `
    -File `
    -Recurse |
    Select-Object -First 1
if (-not $cuda12Runtime) {
    throw "The backend bundle is missing cudart64_12.dll required by eventmamba_selective_scan.dll"
}

Invoke-PackagedBackendSmoke `
    -BackendExecutable (Join-Path $BackendRuntimeDir "UI_Event_Backend.exe") `
    -CenterModel (Join-Path $artifactDestinationDir "eventmamba_center_native_fps.onnx") `
    -EllipseModel (Join-Path $artifactDestinationDir "eventmamba_ellipse_native_fps.onnx") `
    -EllipseMatrix (Join-Path $artifactDestinationDir "eventmamba_ellipse_matrix_A.npy") `
    -CustomOpLibrary $nativeDllDestination

Write-Host "==> Portable bundle ready: $UiBundleDir"

if ($SkipInstaller) {
    Write-Host "==> Skipping installer build because -SkipInstaller was provided"
    exit 0
}

$iscc = Find-InnoCompiler
if (-not $iscc) {
    throw "Inno Setup compiler was not found. Install Inno Setup 6, or set the ISCC environment variable to ISCC.exe."
}

Write-Host "==> Building setup installer"
& $iscc $InstallerScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed"
}

$setupPath = Join-Path $ProjectRoot "installer\UI_Event_Setup.exe"
Assert-FileExists $setupPath "Setup installer"

Write-Host "==> Installer ready: $setupPath"
