@echo off
setlocal
set ROOT=%~dp0
if exist "%ROOT%UI_Event.exe" (
    start "" "%ROOT%UI_Event.exe"
) else (
    start "" "%ROOT%..\dist\UI_Event.exe"
)
