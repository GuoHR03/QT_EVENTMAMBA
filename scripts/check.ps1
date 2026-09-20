[CmdletBinding()]
param(
    [switch]$SkipUiSmoke,
    [switch]$SkipBackendContract
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DevDirectory = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv-dev"))
$DevPython = Join-Path $DevDirectory "Scripts\python.exe"
$UiPython = Join-Path $ProjectRoot ".qtcreator\Pythonvenv\Scripts\python.exe"
$BackendPython = Join-Path $ProjectRoot ".venv-onnx-win\Scripts\python.exe"

function Assert-FileExists {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description is missing: $Path"
    }
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Assert-FileExists $DevPython "Development interpreter; run .\scripts\setup_dev.ps1 first"
$PrefixCheck = @"
import pathlib, platform, sys
expected = pathlib.Path(r'$DevDirectory').resolve()
actual = pathlib.Path(sys.prefix).resolve()
if platform.python_version() != '3.13.5':
    raise SystemExit(f'Expected Python 3.13.5, got {platform.python_version()}')
if actual != expected or sys.prefix == sys.base_prefix:
    raise SystemExit(f'Tests must run from .venv-dev, got {actual}')
print(f'Test interpreter: {sys.executable}')
"@
Invoke-Checked "Verify dedicated test interpreter" $DevPython @("-c", $PrefixCheck)
Invoke-Checked `
    "Validate development dependency contract" `
    $DevPython `
    @("tools\validate_runtime_contract.py", "--role", "test")
Invoke-Checked `
    "Validate UTF-8 source text" `
    $DevPython `
    @("tools\validate_source_text.py")
Invoke-Checked `
    "Validate inference asset manifest" `
    $DevPython `
    @("tools\validate_asset_manifest.py")
Invoke-Checked "Validate Windows inference artifacts" $DevPython @(
    "tools\validate_windows_inference_artifacts.py",
    "--randla-ellipse", "artifacts\eventmamba_ellipse_randla_native.onnx",
    "--randla-matrix", "artifacts\eventmamba_ellipse_randla_matrix_A.npy"
)

$env:PYTHONUTF8 = "1"
$env:QT_QPA_PLATFORM = "offscreen"
$env:UI_EVENT_ALLOW_QT_STUBS = "0"

Invoke-Checked "Verify real Qt import" $DevPython @(
    "-c",
    "from PyQt6 import QtCore; print('Qt mode: real; PyQt6=' + QtCore.PYQT_VERSION_STR)"
)
Invoke-Checked "Run test suite" $DevPython @("-m", "pytest", "-q")
Invoke-Checked "Run performance regression benchmarks" $DevPython @(
    "tools\benchmark_hot_paths.py", "--check"
)
Invoke-Checked "Run type checks" $DevPython @(
    "-m", "mypy",
    "app", "backend", "tools\validate_runtime_contract.py",
    "--ignore-missing-imports", "--no-error-summary"
)

Write-Host "==> Validate PowerShell script syntax"
foreach ($Script in @(
    "scripts\setup_dev.ps1",
    "scripts\check.ps1",
    "scripts\build_installer.ps1",
    "tools\build_selective_scan_ort.ps1"
)) {
    $Tokens = $null
    $Errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        (Resolve-Path -LiteralPath $Script),
        [ref]$Tokens,
        [ref]$Errors
    ) | Out-Null
    if ($Errors.Count) {
        $Errors | Format-List
        throw "PowerShell syntax validation failed: $Script"
    }
}

if (-not $SkipUiSmoke) {
    Assert-FileExists $UiPython "Python 3.8 UI interpreter"
    Invoke-Checked "Validate Python 3.8 UI runtime" $UiPython @(
        "tools\validate_runtime_contract.py", "--role", "ui", "--sdk-root", "libs"
    )
    Invoke-Checked "Smoke-test Metavision runtime" $UiPython @(
        "main.py", "--runtime-smoke-test"
    )
    Invoke-Checked "Construct and close a real MainWindow" $UiPython @(
        "tools\smoke_ui_window.py", "--timeout-ms", "15000"
    )
}

if (-not $SkipBackendContract) {
    Assert-FileExists $BackendPython "Python 3.13 Windows backend interpreter"
    Invoke-Checked "Validate Windows inference runtime" $BackendPython @(
        "tools\validate_runtime_contract.py", "--role", "windows_backend"
    )
}

Write-Host "All requested checks passed."
