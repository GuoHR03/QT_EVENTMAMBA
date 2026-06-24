# 打包安装包说明

本项目使用两步打包：

1. `PyInstaller` 将 Windows 端界面打包成 `dist/UI_Event.exe`。
2. `Inno Setup` 将 exe 和辅助文件打包成 `installer/UI_Event_Setup.exe`。

## 环境要求

- Windows 10/11
- Python 环境中已安装项目运行依赖
- `pyinstaller`
- Inno Setup 6

安装打包工具：

```powershell
pip install pyinstaller
```

Inno Setup 需要从官网安装，安装后脚本会自动寻找 `ISCC.exe`。

## 生成安装包

在项目根目录运行：

```powershell
.\scripts\build_installer.ps1 -Clean
```

成功后会生成：

```text
installer/UI_Event_Setup.exe
```

如果只想生成 exe，不生成安装包：

```powershell
.\scripts\build_installer.ps1 -Clean -SkipInstaller
```

## WSL 后端环境

如果要让安装器自动导入 WSL 推理环境，请先准备：

```text
wsl/eventmamba.tar
```

然后重新运行：

```powershell
.\scripts\build_installer.ps1 -Clean
```

当 `wsl/eventmamba.tar` 存在时，安装器会显示“导入内置 WSL 推理环境”选项，并调用 `scripts/install_wsl.bat` 导入名为 `EventMamba_mini` 的 WSL 发行版。

如果没有该 tar 文件，安装器仍会正常生成，但只安装 Windows 端界面程序。
