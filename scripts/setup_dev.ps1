[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDirectory = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv-dev"))
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$PipTempRoot = Join-Path $VenvDirectory ".pip-tmp"
$RequirementsFile = Join-Path $ProjectRoot "requirements\dev.txt"
$ContractValidator = Join-Path $ProjectRoot "tools\validate_runtime_contract.py"

function Resolve-PythonExecutable {
    param([string]$Candidate)

    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    $command = Get-Command $Candidate -ErrorAction Stop
    return $command.Source
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

$BasePython = Resolve-PythonExecutable $Python
$PythonVersion = (& $BasePython -c "import platform; print(platform.python_version())").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect Python executable: $BasePython"
}
if ($PythonVersion -ne "3.13.5") {
    throw "Development environment requires Python 3.13.5; got $PythonVersion from $BasePython"
}

if ($Recreate -and (Test-Path -LiteralPath $VenvDirectory)) {
    $ExpectedDirectory = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv-dev"))
    if ($VenvDirectory -ne $ExpectedDirectory -or $VenvDirectory -eq $ProjectRoot) {
        throw "Refusing to remove unexpected environment path: $VenvDirectory"
    }
    Write-Host "==> Removing existing development environment: $VenvDirectory"
    Remove-Item -LiteralPath $VenvDirectory -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Invoke-Checked `
        "Create isolated Python 3.13.5 development environment" `
        $BasePython `
        @("-m", "venv", $VenvDirectory)
}

$PrefixCheck = @"
import pathlib, sys
expected = pathlib.Path(r'$VenvDirectory').resolve()
actual = pathlib.Path(sys.prefix).resolve()
if actual != expected or sys.prefix == sys.base_prefix:
    raise SystemExit(f'Not an isolated .venv-dev environment: {actual}')
print(f'Development interpreter: {sys.executable}')
"@
Invoke-Checked "Verify isolated interpreter" $VenvPython @("-c", $PrefixCheck)
if (-not (Test-Path -LiteralPath $PipTempRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $PipTempRoot | Out-Null
}
$PreviousTemp = $env:TEMP
$PreviousTmp = $env:TMP
try {
    # Virus scanners can briefly lock a freshly downloaded Windows wheel.
    # Each retry gets an isolated temp directory so no stale handle can block
    # the next attempt.
    $InstallExitCode = 1
    for ($Attempt = 1; $Attempt -le 5; $Attempt++) {
        $AttemptTemp = Join-Path $PipTempRoot ("attempt-{0}-{1}" -f $Attempt, [guid]::NewGuid())
        New-Item -ItemType Directory -Path $AttemptTemp | Out-Null
        $env:TEMP = $AttemptTemp
        $env:TMP = $AttemptTemp
        Write-Host "==> Install pinned development dependencies (attempt $Attempt/5)"
        & $VenvPython -m pip install `
            --disable-pip-version-check `
            --cache-dir (Join-Path $VenvDirectory ".pip-cache") `
            -r $RequirementsFile
        $InstallExitCode = $LASTEXITCODE
        if ($InstallExitCode -eq 0) {
            break
        }
        if ($Attempt -lt 5) {
            Write-Warning "pip install failed with exit code $InstallExitCode; retrying with a fresh temp directory"
            Start-Sleep -Seconds 2
        }
    }
    if ($InstallExitCode -ne 0) {
        throw "Install pinned development dependencies failed after 5 attempts"
    }
}
finally {
    $env:TEMP = $PreviousTemp
    $env:TMP = $PreviousTmp
}
Invoke-Checked `
    "Validate development dependency contract" `
    $VenvPython `
    @($ContractValidator, "--role", "test")
Invoke-Checked `
    "Verify native Qt import" `
    $VenvPython `
    @("-c", "from PyQt6 import QtCore; print('Qt mode: real; PyQt6=' + QtCore.PYQT_VERSION_STR)")

Write-Host "Development environment is ready: $VenvDirectory"
Write-Host "Run all checks with: .\scripts\check.ps1"
