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
$MetavisionSdkRoot = if ($env:METAVISION_SDK_PATH) {
    [Environment]::ExpandEnvironmentVariables($env:METAVISION_SDK_PATH.Trim().Trim('"'))
} else {
    "E:\Metavision\Prophesee"
}
$MetavisionDestinationRoot = Join-Path $UiBundleDir "metavision"
$MetavisionRuntimeDirectories = @(
    "bin",
    "third_party\bin",
    "lib\hdf5\plugin",
    "lib\metavision\hal\plugins"
)
$MetavisionRequiredFiles = @(
    "third_party\bin\opencv_core4.dll",
    "third_party\bin\opencv_imgproc4.dll",
    "lib\hdf5\plugin\H5Zecf.dll",
    "lib\metavision\hal\plugins\hal_plugin_prophesee.dll",
    "lib\metavision\hal\plugins\metavision_psee_hw_layer.dll"
)
$ArtifactNames = @(
    "eventmamba_center_selective_scan_cuda.onnx",
    "eventmamba_ellipse_selective_scan_cuda.onnx",
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

function Assert-FilePatternExists {
    param(
        [string]$Directory,
        [string]$Pattern,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "$Label directory was not found: $Directory"
    }

    $match = Get-ChildItem `
        -LiteralPath $Directory `
        -Filter $Pattern `
        -File |
        Select-Object -First 1
    if (-not $match) {
        throw "$Label was not found in $Directory (expected $Pattern)"
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
foreach ($runtimeDirectory in $MetavisionRuntimeDirectories) {
    Assert-DirectoryExists `
        (Join-Path $MetavisionSdkRoot $runtimeDirectory) `
        "Required Metavision runtime directory"
}
foreach ($runtimeFile in $MetavisionRequiredFiles) {
    Assert-FileExists `
        (Join-Path $MetavisionSdkRoot $runtimeFile) `
        "Required Metavision runtime file"
}
Assert-FilePatternExists `
    (Join-Path $MetavisionSdkRoot "third_party\bin") `
    "boost_filesystem*.dll" `
    "Required Metavision Boost.Filesystem runtime"

Write-Host "==> UI_Event release $ReleaseVersion"
Write-Host "==> Project: $ProjectRoot"
Write-Host "==> UI Python: $UiPython"
Write-Host "==> Backend Python: $BackendPython"
Write-Host "==> Metavision SDK: $MetavisionSdkRoot"

Assert-PyInstaller $UiPython "UI environment"
Assert-PyInstaller $BackendPython "Windows inference environment"

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
New-Item -ItemType Directory -Path $BackendRuntimeDir | Out-Null
Get-ChildItem -LiteralPath $BackendBundleDir -Force |
    Copy-Item -Destination $BackendRuntimeDir -Recurse -Force

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
foreach ($runtimeDirectory in $MetavisionRuntimeDirectories) {
    $runtimeSource = Join-Path $MetavisionSdkRoot $runtimeDirectory
    $runtimeDestination = Join-Path $MetavisionDestinationRoot $runtimeDirectory
    New-Item `
        -ItemType Directory `
        -Path (Split-Path -Parent $runtimeDestination) `
        -Force |
        Out-Null
    Copy-Item `
        -LiteralPath $runtimeSource `
        -Destination $runtimeDestination `
        -Recurse `
        -Force
}

$requiredBundleFiles = @(
    $uiExePath,
    (Join-Path $BackendRuntimeDir "UI_Event_Backend.exe"),
    (Join-Path $artifactDestinationDir "eventmamba_center_selective_scan_cuda.onnx"),
    (Join-Path $artifactDestinationDir "eventmamba_ellipse_selective_scan_cuda.onnx"),
    (Join-Path $artifactDestinationDir "eventmamba_ellipse_matrix_A.npy"),
    $nativeDllDestination,
    (Join-Path $UiBundleDir "_internal\app\form.ui"),
    (Join-Path $UiBundleDir "_internal\app\choose_form.ui")
)
foreach ($requiredFile in $requiredBundleFiles) {
    Assert-FileExists $requiredFile "Required release file"
}
Assert-DirectoryExists (Join-Path $UiBundleDir "_internal\libs") "Bundled Metavision libraries"
Assert-DirectoryExists (Join-Path $BackendRuntimeDir "_internal") "Bundled Windows inference dependencies"
foreach ($runtimeDirectory in $MetavisionRuntimeDirectories) {
    Assert-DirectoryExists `
        (Join-Path $MetavisionDestinationRoot $runtimeDirectory) `
        "Staged Metavision runtime directory"
}
foreach ($runtimeFile in $MetavisionRequiredFiles) {
    Assert-FileExists `
        (Join-Path $MetavisionDestinationRoot $runtimeFile) `
        "Staged Metavision runtime file"
}
Assert-FilePatternExists `
    (Join-Path $MetavisionDestinationRoot "third_party\bin") `
    "boost_filesystem*.dll" `
    "Staged Metavision Boost.Filesystem runtime"

$cuda12Runtime = Get-ChildItem `
    -LiteralPath (Join-Path $BackendRuntimeDir "_internal") `
    -Filter "cudart64_12.dll" `
    -File `
    -Recurse |
    Select-Object -First 1
if (-not $cuda12Runtime) {
    throw "The backend bundle is missing cudart64_12.dll required by eventmamba_selective_scan.dll"
}

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
