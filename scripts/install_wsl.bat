@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%wsl" if exist "%ROOT%..\wsl" for %%I in ("%ROOT%..") do set "ROOT=%%~fI\"
set "WSL_DIR=%ROOT%wsl"
set "TARBALL=%WSL_DIR%\eventmamba.tar"
set "DISTRO=EventMamba_mini"
set "LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python"
set "NEEDS_RESTART=0"

if not exist "%WSL_DIR%" mkdir "%WSL_DIR%"
if not exist "%TARBALL%" echo Missing %TARBALL% && exit /b 2
where wsl.exe >nul 2>nul
if errorlevel 1 echo wsl.exe was not found && exit /b 3

dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if %ERRORLEVEL% EQU 3010 set NEEDS_RESTART=1
if %ERRORLEVEL% NEQ 0 if %ERRORLEVEL% NEQ 3010 exit /b 4

dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if %ERRORLEVEL% EQU 3010 set NEEDS_RESTART=1
if %ERRORLEVEL% NEQ 0 if %ERRORLEVEL% NEQ 3010 exit /b 5

if "%NEEDS_RESTART%"=="1" (
    echo Windows features were enabled and a restart is required before importing WSL.
    exit /b 10
)

wsl --set-default-version 2
if errorlevel 1 exit /b 11

wsl --unregister %DISTRO% 2>nul
wsl --import %DISTRO% "%WSL_DIR%\%DISTRO%" "%TARBALL%"
if errorlevel 1 exit /b 12

wsl -d %DISTRO% %LINUX_PYTHON% --version >nul 2>nul
if errorlevel 1 exit /b 13

wsl -d %DISTRO% %LINUX_PYTHON% -c "import torch, zmq; print('ok')" >nul 2>nul
if errorlevel 1 exit /b 14

echo WSL import completed
exit /b 0
