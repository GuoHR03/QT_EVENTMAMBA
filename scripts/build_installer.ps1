param(
    [switch]$Clean,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot ".qtcreator\Pythonvenv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project Python interpreter was not found: $ProjectPython"
}

function Remove-GeneratedPath {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolvedTarget = Resolve-Path -LiteralPath $Path
    $resolvedRoot = Resolve-Path -LiteralPath $ProjectRoot
    if (-not $resolvedTarget.Path.StartsWith($resolvedRoot.Path)) {
        throw "Refusing to remove path outside project: $($resolvedTarget.Path)"
    }

    Remove-Item -LiteralPath $resolvedTarget.Path -Recurse -Force
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
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 5\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 5\ISCC.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

Write-Host "==> Project: $ProjectRoot"
Write-Host "==> Python: $ProjectPython"

if ($Clean) {
    Write-Host "==> Cleaning generated build outputs"
    Remove-GeneratedPath (Join-Path $ProjectRoot "build")
    Remove-GeneratedPath (Join-Path $ProjectRoot "dist")
    Remove-GeneratedPath (Join-Path $ProjectRoot "installer")
}

Write-Host "==> Checking PyInstaller"
& $ProjectPython -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in the project environment. Install it with: `"$ProjectPython`" -m pip install pyinstaller"
}

Write-Host "==> Building UI_Event.exe"
& $ProjectPython -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "UI_Event.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed"
}

$exePath = Join-Path $ProjectRoot "dist\UI_Event.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Expected executable was not created: $exePath"
}
Write-Host "==> EXE ready: $exePath"

if ($SkipInstaller) {
    Write-Host "==> Skipping installer build because -SkipInstaller was provided"
    exit 0
}

$iscc = Find-InnoCompiler
if (-not $iscc) {
    throw "Inno Setup compiler was not found. Install Inno Setup 6, or set the ISCC environment variable to ISCC.exe."
}

Write-Host "==> Building setup installer"
& $iscc (Join-Path $ProjectRoot "installer.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed"
}

$setupPath = Join-Path $ProjectRoot "installer\UI_Event_Setup.exe"
if (-not (Test-Path -LiteralPath $setupPath)) {
    throw "Expected installer was not created: $setupPath"
}

Write-Host "==> Installer ready: $setupPath"
