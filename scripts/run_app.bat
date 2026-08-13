@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "APP_EXE="

if exist "%SCRIPT_DIR%UI_Event.exe" (
    set "APP_EXE=%SCRIPT_DIR%UI_Event.exe"
)

if not defined APP_EXE if exist "%SCRIPT_DIR%..\dist\UI_Event\UI_Event.exe" (
    set "APP_EXE=%SCRIPT_DIR%..\dist\UI_Event\UI_Event.exe"
)

if not defined APP_EXE (
    echo UI_Event.exe was not found.
    echo Expected an installed copy beside this script or a development build at dist\UI_Event\UI_Event.exe.
    exit /b 1
)

start "" "%APP_EXE%"
