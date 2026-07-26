#define MyAppVersion "0.2.0"

[Setup]
AppId={{A9C7D7D0-6B94-4B50-9F6F-1E7E5A0B2C31}
AppName=事件相机推理工具
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}.0
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
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\UI_Event.exe

[Files]
Source: "dist\UI_Event\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "scripts\run_app.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "PACKAGING.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "VERSION.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\事件相机推理工具"; Filename: "{app}\UI_Event.exe"
Name: "{commondesktop}\事件相机推理工具"; Filename: "{app}\UI_Event.exe"
