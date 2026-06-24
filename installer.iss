[Setup]
AppId={{A9C7D7D0-6B94-4B50-9F6F-1E7E5A0B2C31}
AppName=事件相机推理工具
AppVersion=1.0.0
AppPublisher=EventMamba
DefaultDirName={autopf}\UI_Event
DefaultGroupName=事件相机推理工具
OutputDir=installer
OutputBaseFilename=UI_Event_Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
ShowLanguageDialog=no
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=yes
DisableReadyMemo=no
DisableReadyPage=no
DisableFinishedPage=no
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\UI_Event.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "scripts\run_app.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "scripts\install_wsl.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "scripts\uninstall_wsl.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "VERSION.md"; DestDir: "{app}"; Flags: ignoreversion
#ifexist "wsl\eventmamba.tar"
Source: "wsl\eventmamba.tar"; DestDir: "{app}\wsl"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\事件相机推理工具"; Filename: "{app}\UI_Event.exe"
Name: "{commondesktop}\事件相机推理工具"; Filename: "{app}\UI_Event.exe"

#ifexist "wsl\eventmamba.tar"
[Tasks]
Name: "installwsl"; Description: "导入内置 WSL 推理环境（首次安装推荐勾选）"; Flags: checkedonce
#endif

[Dirs]
Name: "{app}\wsl"

[Code]
#ifexist "wsl\eventmamba.tar"
function RunBundledWslInstall(): Boolean;
var
  ResultCode: Integer;
begin
  Result := False;

  if not Exec(ExpandConstant('{cmd}'),
    '/C ""' + ExpandConstant('{app}\install_wsl.bat') + '""',
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode) then
  begin
    MsgBox('无法启动 WSL 推理环境安装脚本，请检查 install_wsl.bat 是否存在且可执行。', mbCriticalError, MB_OK);
    Exit;
  end;

  case ResultCode of
    0:
      begin
        MsgBox('WSL 推理环境安装完成。', mbInformation, MB_OK);
        Result := True;
      end;
    2:
      MsgBox('未找到内置的 WSL 推理环境镜像 eventmamba.tar。', mbCriticalError, MB_OK);
    3:
      MsgBox('系统中未找到 wsl.exe。请先安装或启用 Windows Subsystem for Linux。', mbCriticalError, MB_OK);
    4, 5:
      MsgBox('启用 WSL 所需的 Windows 功能时失败，请以管理员身份重新运行安装程序。', mbCriticalError, MB_OK);
    10:
      MsgBox('已启用 WSL 所需的 Windows 功能，但需要先重启电脑。请重启后重新运行安装程序完成推理环境安装。', mbInformation, MB_OK);
    11:
      MsgBox('无法将 WSL 默认版本设置为 2，请确认系统支持 WSL2。', mbCriticalError, MB_OK);
    12:
      MsgBox('导入 EventMamba_mini 发行版失败，请确认磁盘空间充足且安装目录可写。', mbCriticalError, MB_OK);
    13:
      MsgBox('已导入 WSL 环境，但未找到 Linux 侧 Python 解释器。请检查导出的环境是否完整。', mbCriticalError, MB_OK);
    14:
      MsgBox('已导入 WSL 环境，但推理依赖自检失败。请检查 Linux 环境中的 torch/zmq 是否完整。', mbCriticalError, MB_OK);
  else
    MsgBox(
      Format('WSL 推理环境安装失败，退出码: %d。请联系开发者排查。', [ResultCode]),
      mbCriticalError,
      MB_OK
    );
  end;
end;
#endif

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssDone then
    Exit;

  #ifexist "wsl\eventmamba.tar"
  if WizardIsTaskSelected('installwsl') then
    RunBundledWslInstall();
  #endif
end;
