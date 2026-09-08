# Windows 打包与安装说明

最近正式版本为 `0.2.1`，包含推理生命周期、严格 ZeroMQ 协议和 native-FPS
打包改动，详见 [CHANGELOG.md](CHANGELOG.md)。默认发布链路完全运行在
Windows 上，不需要 WSL，也不要求
目标电脑安装 Python。

发布包采用两个独立的 PyInstaller `onedir` 程序：

1. `.qtcreator/Pythonvenv` 构建图形界面到 `dist/UI_Event/UI_Event.exe`。
2. `.venv-onnx-win` 构建 ONNX/CUDA 推理后端到 `dist/UI_Event_Backend/`。
3. 构建脚本把后端目录移动到 `dist/UI_Event/backend_runtime/`，并暂存三个正式推理资源、自定义算子和经过白名单筛选的 Metavision Runtime。
4. Inno Setup 递归打包 `dist/UI_Event/`，生成 `installer/UI_Event_Setup.exe`。

`onedir` 是有意选择的发布形式。界面、推理后端、CUDA 运行库和模型资源都保留稳定的目录结构，避免单文件模式每次启动解压到临时目录所造成的路径和子进程问题。

## 构建环境

发布工作分成两条路径：

1. **使用仓库中已有的正式资产打包**：适合日常生成便携目录或安装程序，
   不需要重新导出 ONNX 或编译自定义算子。
2. **重新生成模型和自定义算子**：仅在模型图、原生算子或 ONNX Runtime
   契约变化时执行，完成后必须重新跑等价性和资产校验。

### 使用已有正式资产打包

需要准备：

- Windows 10/11 x64。
- `.qtcreator/Pythonvenv/Scripts/python.exe`（Python 3.8.20 x64），已按 `requirements/ui-py38.txt` 安装 UI 运行依赖；正式包中的 Metavision 原生扩展固定为 CPython 3.8 ABI。
- `.venv-onnx-win/Scripts/python.exe`（Python 3.13.5），已按 `requirements/backend-py313.txt` 安装 ONNX/CUDA 后端依赖，并通过 CUDA 运行库安装脚本补齐 NVIDIA Wheel。
- Inno Setup 6；如果 `ISCC.exe` 不在常见安装目录或 `PATH` 中，可设置 `ISCC` 环境变量。
- 项目 `libs/` 中的 Metavision Runtime。构建脚本优先读取 `METAVISION_SDK_PATH`，未设置时优先使用项目内运行库；运行库不完整时才回退到 `E:\Metavision\Prophesee`。
- 下列正式推理文件：

```text
artifacts/eventmamba_center_native_fps.onnx
artifacts/eventmamba_ellipse_native_fps.onnx
artifacts/eventmamba_ellipse_matrix_A.npy
native/selective_scan_ort/bin/eventmamba_selective_scan.dll
```

### 重新生成模型和自定义算子

只有修改模型图或 `native/selective_scan_ort/` 后才需要执行本节。除上述两个
Python 环境外，还需要 Visual Studio 2022 C++ 工具链、CMake、Ninja、CUDA
Toolkit 12.2 和 ONNX Runtime 1.27 开发头文件。

`tools/build_selective_scan_ort.ps1` 会通过 `vswhere.exe`、`PATH`、Visual
Studio自带工具目录和 `CUDA_PATH_V12_2` 自动寻找构建工具。可以先执行
`.\tools\build_selective_scan_ort.ps1 -ResolveOnly` 查看最终路径而不产生构建
输出；非标准安装可通过 `-VsInstallPath`、`-CMakePath`、`-NinjaPath`、
`-CudaRoot` 和 `-OrtIncludeDir` 显式覆盖。

模型与 DLL 必须来自同一套原生 FPS 契约。重建顺序为：

```powershell
.\tools\build_selective_scan_ort.ps1
.\.venv-onnx-win\Scripts\python.exe tools\onnx_insert_hierarchical_fps.py `
  --input artifacts\eventmamba_center_selective_scan_cuda.onnx `
  --output artifacts\eventmamba_center_native_fps.onnx `
  --overwrite
.\.venv-onnx-win\Scripts\python.exe tools\onnx_insert_hierarchical_fps.py `
  --input artifacts\eventmamba_ellipse_selective_scan_cuda.onnx `
  --output artifacts\eventmamba_ellipse_native_fps.onnx `
  --overwrite
.\.venv-onnx-win\Scripts\python.exe tools\validate_windows_inference_artifacts.py
.\.venv-onnx-win\Scripts\python.exe tools\onnx_native_fps_equivalence_probe.py
```

最后一项需要 NVIDIA GPU；它用相同事件和 FPS 起点比较新旧 center/ellipse
输出。安装包构建还会自动执行静态资产校验和 CPU 原生 FPS 算子探针，防止
新模型与旧 DLL 混装。

创建或恢复两个发布环境时，分别执行：

```powershell
.\.qtcreator\Pythonvenv\Scripts\python.exe -m pip install -r requirements\ui-py38.txt
.\.venv-onnx-win\Scripts\python.exe -m pip install -r requirements\backend-py313.txt
.\tools\install_onnx_cuda_runtime.ps1
```

两个环境不能互换：UI 环境负责 PyQt6 和 Metavision，后端环境负责 ONNX Runtime GPU、NumPy、ZeroMQ、CUDA 与 cuDNN 运行依赖。

`runtime-contract.json` 是发布运行时的机器可读契约，记录两个解释器、关键
Python 分发包、Metavision SDK/ABI 以及原生算子的 ORT/CUDA 构建契约。
`requirements.txt` 保留为 UI 依赖入口；Windows 后端必须使用独立清单。
可在不打包的情况下单独检查环境：

```powershell
.\.qtcreator\Pythonvenv\Scripts\python.exe tools\validate_runtime_contract.py `
  --role ui --sdk-root libs
.\.venv-onnx-win\Scripts\python.exe tools\validate_runtime_contract.py `
  --role windows_backend
```

`scripts/build_installer.ps1` 会在 PyInstaller 启动前自动运行相同检查，任何
解释器、关键依赖、SDK 版本或 CPython 扩展漂移都会使发布构建立即失败。

自定义算子当前直接依赖 CUDA 12 的 `cudart64_12.dll`。后端 spec 会从 `CUDA_PATH_V12_2` 或标准 CUDA 12.2 安装目录收集该 DLL；构建脚本也会在生成安装器前检查后端冻结目录中确实存在它。

为了保留实时相机、RAW/H5 和 HAL 插件能力，构建会从 Metavision SDK 精确收集所需的第三方 Release DLL、HDF5 插件和 HAL 插件。不会复制 SDK 的 CLI 示例、Debug DLL、导入库或无关模块。所需目录为：

```text
third_party/bin/
lib/hdf5/plugin/
lib/metavision/hal/plugins/
```

如需改用其他目录中的 SDK Runtime，先设置：

```powershell
$env:METAVISION_SDK_PATH = "D:\Path\To\Prophesee"
```

任一必需目录缺失时，构建会在清理旧产物之前失败，避免误生成缺少相机功能的安装包。

## 生成发布包

在项目根目录运行：

```powershell
.\scripts\build_installer.ps1 -Clean
```

成功后会生成：

```text
installer/UI_Event_Setup.exe
```

只构建完整便携目录、不调用 Inno Setup：

```powershell
.\scripts\build_installer.ps1 -Clean -SkipInstaller
```

同时生成适合上传到 GitHub Releases 的便携 ZIP：

```powershell
.\scripts\build_installer.ps1 -Clean -SkipInstaller -PortableArchive
```

ZIP 输出为 `installer/UI_Event-<version>-windows-x64-portable.zip`，构建脚本会
在结束时打印文件的 SHA-256，便于下载后核验。

输出目录为：

```text
dist/UI_Event/
├── UI_Event.exe
├── _internal/
│   ├── app/
│   │   └── form.ui
│   └── libs/bin/              # 仅所需 Metavision Release DLL
├── backend_runtime/
│   ├── UI_Event_Backend.exe
│   └── _internal/
├── artifacts/
│   ├── eventmamba_center_native_fps.onnx
│   ├── eventmamba_ellipse_native_fps.onnx
│   └── eventmamba_ellipse_matrix_A.npy
├── metavision/
│   ├── third_party/bin/
│   ├── lib/hdf5/plugin/
│   └── lib/metavision/hal/plugins/
└── native/selective_scan_ort/bin/
    └── eventmamba_selective_scan.dll
```

请整体复制 `dist/UI_Event/`，不能只复制两个 EXE。两个 `_internal/` 目录以及其中的 DLL、Python 包和 Qt/ONNX Runtime 文件都是运行所必需的。

后端构建产物通过同卷 `Move-Item` 进入 `backend_runtime/`；成功后不会额外保留一份 `dist/UI_Event_Backend/`。构建脚本在调用安装器前会验证以下内容：

- UI EXE、界面 `.ui` 文件和 Metavision 库目录。
- 后端 EXE 及其完整 `_internal` 目录。
- 两个 ONNX 模型、椭圆解码矩阵和自定义算子 DLL。
- ONNX/DLL 原生 FPS 契约、独立算子逐项等价探针，以及冻结后端的
  `center -> ellipse -> center` 实际预测冒烟。
- 自定义算子依赖的 `cudart64_12.dll`。
- CPython 3.8 Metavision 扩展、主 Release DLL、白名单第三方运行库、HDF5 插件和 HAL 插件。
- `opencv_core4.dll`、`opencv_imgproc4.dll`、Boost.Filesystem、`metavision_psee_hw_layer.dll`、`H5Zecf.dll` 和 `hal_plugin_prophesee.dll` 等关键相机运行文件。

缺少任意必需文件都会立即终止，不会生成一个已知不完整的安装包。

发布包不会携带仓库中的 `backend/` 源码副本、`__pycache__`、Metavision Debug/CPython 3.9 扩展或 SDK 命令行示例。Python 模块由 PyInstaller 收入 PYZ，仅保留运行所需的 CPython 3.8 原生扩展。

`-Clean` 只允许删除项目根目录下精确命名的 `build/`、`dist/` 和 `installer/` 生成目录，不会触碰源码、模型源文件、录像或用户数据。

## 目标电脑要求

安装版已经包含两个 Python 冻结运行时、ONNX Runtime GPU、构建环境中收集到的 CUDA/cuDNN DLL、模型、自定义算子和 Metavision 用户态运行库。目标电脑通常不需要安装 Python、WSL、CUDA Toolkit、cuDNN 或完整 Metavision SDK，但仍需要：

- Windows 10/11 x64。
- 与打包运行库兼容的 NVIDIA GPU 和 NVIDIA 驱动。
- Microsoft Visual C++ 2015–2022 x64 Runtime。
- 使用事件相机时所需的 Metavision 设备驱动；实时相机功能还需要受支持的硬件。HAL 插件虽已随包携带，但不能代替内核/USB 设备驱动。

当前 GPU 支持状态：

| 项目 | 状态 |
| --- | --- |
| NVIDIA CUDA 架构 7.5 | 原生 DLL 构建目标，仍需在具体显卡上验收 |
| NVIDIA CUDA 架构 8.6 | 原生 DLL 构建目标，仍需在具体显卡上验收 |
| 其他 NVIDIA 架构 | 未完整验证；可能需要重新编译自定义算子 |
| Windows CPU 推理 | 未完成正式回退验证 |
| 无 NVIDIA GPU 的显示/回放 | 可用，但 EventMamba 原生推理不可保证 |

正式交付前必须在没有开发环境的目标机上做完整冒烟测试，确认自定义算子的 CUDA 12 Runtime 和驱动兼容性。

## WSL 源码兼容

0.2.1 安装器不再携带 `wsl/eventmamba.tar`，也不会安装、启用或导入 WSL。默认安装包始终使用 `backend_runtime/UI_Event_Backend.exe`。

源码运行方式仍保留原 WSL 兼容后端。需要对照旧 PyTorch/Mamba 链路时，可在自行准备好 WSL2 和 `EventMamba_mini` 环境后设置：

```text
EVENTMAMBA_INFERENCE_RUNTIME=wsl
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
```

仓库中的 `scripts/install_wsl.bat` 和 `scripts/uninstall_wsl.bat` 仅用于手动兼容流程，不进入默认安装包。

## 发布前验收

至少在一台未安装 Python、Qt、开发版 CUDA/cuDNN 和 WSL 的干净 Windows 电脑上验证：

- 安装、启动、升级和卸载。
- RAW 回放、进度、seek、倍速及自然结束后重新播放；H5/HDF5、AEDAT4 作为兼容路径做基础打开冒烟。
- 实时相机连接、ROI、去噪和录制，并确认实时模式不显示 seek、进度或倍速控件。
- `center` 与 `ellipse` 模型启动、停止、重启和 CUDA 推理；连续循环后确认端口与后端进程均已释放。
- 人工终止后端或占用推理端口时，界面应进入错误状态，旧实例不得通过健康检查冒充新实例。
- 中文路径、带空格路径和非管理员用户运行。
- 日志与录像写入用户数据目录，而不是 `Program Files`。

安装器不会卸载或修改用户数据目录中的日志、录像和个人文件。
