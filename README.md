# UI_Event

UI_Event 是一个基于 PyQt6 的事件相机可视化与 EventMamba 推理工具。它可以连接实时 Metavision 相机，也可以回放 RAW、H5/HDF5 和 AEDAT4 事件文件，并在画面上叠加中心点或椭圆预测结果。

当前正式运行链路以 Windows 原生 ONNX Runtime CUDA 后端为主。安装版不需要目标电脑安装 Python、WSL、CUDA Toolkit 或 cuDNN；WSL/PyTorch 后端仅作为源码兼容与结果对照方案保留。

## 主要功能

- 实时 Metavision 相机预览和 RAW 数据录制。
- RAW、H5/HDF5、AEDAT4 离线回放，支持进度、拖拽定位和播放倍速。
- `Dark`、`Light`、`CoolWarm`、`Gray` 四种事件调色板。
- Activity、Trail、STC、AntiFlicker 去噪。
- 实时相机硬件 ROI 尝试配置，以及离线文件软件 ROI 热更新。
- EventMamba `center` 中心点预测和 `ellipse` 椭圆预测。
- 预测结果按事件时间戳与图像帧匹配后叠加显示。
- Windows 原生 ONNX/CUDA 推理，以及可选的 WSL/PyTorch 兼容后端。

## 输入能力

| 输入源 | 显示 | 进度/Seek | 倍速 | ROI | 去噪 | 推理 | 录制 | 额外要求 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 实时 Metavision 相机 | 是 | 不适用 | 不适用 | 硬件 ROI 优先 | 是 | 是 | RAW | 相机及设备驱动 |
| RAW | 是 | 是 | 是 | 软件 ROI | 是 | 是 | 否 | Metavision SDK/运行库 |
| H5/HDF5 | 是 | 是 | 是 | 软件 ROI | 是 | 是 | 否 | `h5py`、Metavision 帧生成组件 |
| AEDAT4 | 是 | 是 | 是 | 软件 ROI | 是 | 是 | 否 | `dv_processing` |

RAW 是主要离线格式；H5/HDF5 和 AEDAT4 作为兼容输入保留。界面按钮仍显示“选择 RAW 文件”，但文件对话框可以选择这三类离线格式。

说明：RAW 和 AEDAT4 可以向底层 reader 转发主动停止；H5 会在当前读取或回放等待结束后响应停止，不保证立即唤醒。

## 5 分钟快速开始

### 使用已经构建好的 Windows 版本

1. 安装 `UI_Event_Setup.exe`，或完整解压/复制便携目录 `dist/UI_Event/`。
2. 启动 `UI_Event.exe`。便携版不能只复制 EXE，必须保留同目录的 `_internal/`、`backend_runtime/`、`artifacts/`、`metavision/` 和 `native/`。
3. 点击画面右下角的设置按钮，展开设置面板。
4. 选择“实时相机”或“选择 RAW 文件”，然后启动相机或文件回放。
5. 如需推理，选择“中心点”或“椭圆”模式，再选择匹配的 ONNX 模型并启动推理服务。
6. 模型启动成功后，预测结果会按时间戳叠加在事件画面上；详细状态和错误显示在状态条及日志区。

安装版运行要求：

- Windows 10/11 x64。
- 兼容的 NVIDIA GPU 和驱动；当前原生 DLL 构建目标为 CUDA 架构 7.5 和 8.6，其他架构尚未完整验证。
- Microsoft Visual C++ 2015–2022 x64 Runtime。
- 使用实时相机时，需要相应的 Metavision 设备驱动和受支持硬件。

目前没有经过完整验证的 Windows CPU 推理回退；没有兼容 NVIDIA GPU 时，事件显示和文件回放仍可使用，但原生 EventMamba 推理不可保证可用。

### 从当前源码工作区启动

项目采用两个独立 Python 环境，不能混用：

- UI/Metavision 环境：发布构建固定使用 CPython 3.8 x64。
- Windows 推理环境：当前验证环境为 Python 3.13，负责 ONNX Runtime GPU、NumPy、ZeroMQ 和 CUDA 运行依赖。

如果两个环境和模型资产已经准备好，可在项目根目录运行：

```powershell
.\.qtcreator\Pythonvenv\Scripts\python.exe main.py
```

UI 基础依赖包括 PyQt6、pyzmq、NumPy 和 h5py；AEDAT4 兼容输入还需要单独安装 `dv_processing`。Windows 推理环境需要 ONNX Runtime GPU，以及与模型配套的 CUDA/cuDNN 运行库和自定义算子 DLL。

当前源码仍有一个需要在发布前处理的兼容性限制：发布 UI 环境固定为 Python 3.8，但部分推理模块包含 Python 3.10 才支持的类型标注。因此不要把 UI 与推理依赖合并到同一个 Python 3.8 环境，也不要把“Python 3.8+”理解为所有源码模块都已在任意高版本上验证。

完整构建环境、资产准备和安装包验收流程见 [PACKAGING.md](PACKAGING.md)。

## 使用说明

### 输入与回放

- 实时相机直接读取事件流，不显示进度、Seek 或倍速控件。
- RAW 按事件时间戳进行真实时间回放，并优先从 `.raw.tmp_index` 获取总时长。
- H5/HDF5 自动识别支持的事件数据集和字段。
- AEDAT4 使用 `dv_processing` 读取事件批次。
- 切换输入源或 Seek 时，旧画面、旧进度和在途推理响应会被丢弃，避免混入新时间点。
- Palette、显示 FPS 和离线播放倍速可以在播放过程中更新，不需要重新打开文件。

显示 FPS 决定事件帧的切分和生成节奏；播放倍速只改变离线文件的时间推进速度。推理使用独立的 20 ms 事件窗口，不直接复用显示帧窗口。

### ROI 与去噪

ROI 使用 `X、Y、宽度、高度` 四个整数：

- 坐标原点位于图像左上角，X 向右、Y 向下。
- X、Y 必须大于等于 0，宽度和高度必须大于 0。
- 四个输入框全部留空并应用设置，可清除 ROI。
- ROI 与当前图像无交集时，界面会拒绝应用。
- 实时相机会优先尝试硬件 ROI；离线文件使用软件 ROI，可在播放中热更新。

去噪发生在显示和推理之前，阈值单位为微秒（μs）。算法选择、阈值方向和推荐范围见 [METAVISION_DENOISE.md](METAVISION_DENOISE.md)。

### 推理模式与输出

| 模式 | 输出 | 含义 |
| --- | --- | --- |
| `center` | `[x, y]` | 目标中心位置 |
| `ellipse` | `[x, y, a, b, angle]` | 中心位置、长半轴、短半轴、旋转角 |

当前模型通常输出归一化坐标；UI 会结合输入分辨率和生效 ROI 映射回像素位置。椭圆的 `a`、`b` 是归一化半轴，`angle` 使用弧度，绘制时转换为角度。

Windows UI 通过本机 ZeroMQ 调用独立推理进程。安装版启动冻结后的 `backend_runtime/UI_Event_Backend.exe`；源码模式启动独立的 Windows 推理 Python。两者均使用 ONNX Runtime CUDA 和项目自定义算子完成 EventMamba 推理。

### RAW 录制

录制仅对实时相机开放，离线文件回放时不可录制。文件名格式为：

```text
recording_YYYYMMDD_HHMMSS.raw
```

- 源码模式默认写入项目的 `record/` 目录。
- 安装版默认写入 `%LOCALAPPDATA%\UI_Event\record`。

卸载程序不会删除用户录制文件和日志。

## Windows 推理资产

Windows 原生推理需要以下四项配套资产：

```text
artifacts/eventmamba_center_native_fps.onnx
artifacts/eventmamba_ellipse_native_fps.onnx
artifacts/eventmamba_ellipse_matrix_A.npy
native/selective_scan_ort/bin/eventmamba_selective_scan.dll
```

这些是仓库跟踪的正式运行资产。`*_selective_scan_cuda.onnx` 是重新生成 native-FPS 模型时使用的源资产；其他实验 ONNX、NPZ、日志和构建目录属于本地产物。

默认模型把三级最远点采样（Farthest Point Sampling，FPS）放入 `com.eventmamba::HierarchicalFarthestPointSampling` 原生算子。模型、椭圆矩阵和 DLL 必须配套更新。

WSL 源码兼容模式使用 PyTorch 权重；`ellipse` 权重同目录还需要 `matrix_A.pt`：

```text
checkpoint/
└── your_model/
    ├── P3best_checkpoint.pth
    └── matrix_A.pt
```

## 环境变量

Windows 是默认推理运行时，通常不需要设置第一项：

```text
EVENTMAMBA_INFERENCE_RUNTIME=windows
EVENTMAMBA_WINDOWS_PYTHON=.venv-onnx-win/Scripts/python.exe
EVENTMAMBA_WINDOWS_BACKEND_EXECUTABLE=backend_runtime/UI_Event_Backend.exe
EVENTMAMBA_CENTER_ONNX_MODEL=artifacts/eventmamba_center_native_fps.onnx
EVENTMAMBA_ELLIPSE_ONNX_MODEL=artifacts/eventmamba_ellipse_native_fps.onnx
EVENTMAMBA_ELLIPSE_MATRIX=artifacts/eventmamba_ellipse_matrix_A.npy
EVENTMAMBA_SELECTIVE_SCAN_DLL=native/selective_scan_ort/bin/eventmamba_selective_scan.dll
EVENTMAMBA_BACKEND_READY_TIMEOUT_S=180
METAVISION_SDK_PATH=E:\Metavision\Prophesee
```

源码兼容的 WSL 后端可使用：

```text
EVENTMAMBA_INFERENCE_RUNTIME=wsl
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
```

UI 与推理后端使用 `eventmamba/v1` 协议：控制消息为受限 JSON，事件数据为固定 `(1024, 3)` little-endian `float32` 二进制帧。更新协议后必须同时替换 UI 与后端，不能混用新旧可执行文件。

## 构建 Windows 发布版

构建便携目录：

```powershell
.\scripts\build_installer.ps1 -Clean -SkipInstaller
```

构建安装程序：

```powershell
.\scripts\build_installer.ps1 -Clean
```

安装包输出为 `installer/UI_Event_Setup.exe`。完整依赖、目录结构、资产验证和干净电脑验收项目见 [PACKAGING.md](PACKAGING.md)。

## 常见问题

- **无法识别实时相机：**检查设备驱动、USB 连接和 Metavision HAL 插件，并查看界面日志。
- **无法打开 AEDAT4：**确认当前 UI 环境已安装 `dv_processing`。
- **推理服务无法启动：**检查 GPU/驱动、两个 ONNX 模型、椭圆矩阵和自定义算子 DLL 是否齐全且来自同一构建版本。
- **模型已启动但没有预测：**确认回放或相机正在产生事件，ROI 内事件数量足够，并检查去噪是否过强。
- **H5 停止稍有延迟：**当前 H5 路径在读取或回放等待边界响应停止，不是底层可唤醒 reader。
- **中文显示乱码：**源码和文档使用 UTF-8；PowerShell 可使用 `Get-Content -Encoding UTF8` 查看。

安装版日志位于 `%LOCALAPPDATA%\UI_Event`。源码模式的后端日志默认为项目运行目录下的 `eventmamba_backend.log`。

## 开发者概览

- `app/`：PyQt6 界面、用户操作、状态显示和预测叠加。
- `backend/`：输入源、事件管线、回放、录制、推理服务和通信协议。
- `artifacts/`：正式 ONNX/矩阵资产及本地实验产物。
- `native/selective_scan_ort/`：Windows ONNX Runtime 自定义算子。
- `scripts/`：安装、运行和打包脚本。
- `tools/`：模型转换、等价验证、性能测试和诊断工具。
- `tests/`：协议、输入源、播放、推理、UI 状态和打包逻辑测试。

推理服务使用 nonce/PID 标识每次后端实例，WSL 停止流程只处理经过命令行 nonce 验证的 PID，不会扫描或广域终止其他任务。切源、Seek 和重启通过请求代际隔离旧画面与旧响应。

## 文档导航

- [CHANGELOG.md](CHANGELOG.md)：正式版本与当前未发布改动。
- [VERSION.md](VERSION.md)：早期内部开发快照历史，不代表正式版本号。
- [PACKAGING.md](PACKAGING.md)：Windows 打包、安装和发布验收。
- [METAVISION_DENOISE.md](METAVISION_DENOISE.md)：去噪算法、阈值和调参建议。
- [ONNX_FEASIBILITY.md](ONNX_FEASIBILITY.md)：Windows 原生 ONNX/CUDA 可行性和性能验证。
- [native/selective_scan_ort/README.md](native/selective_scan_ort/README.md)：自定义算子约束与组件开发。

## 当前验证与已知限制

- 当前自动化测试：`384 passed`。
- Windows 原生推理只验证了 NVIDIA/CUDA 路径，没有完整验证 CPU 回退。
- 原生 DLL 当前构建目标为 CUDA 架构 7.5 和 8.6，其他 NVIDIA 架构需要额外验证或重新编译。
- 发布 UI 固定使用 CPython 3.8 ABI，但部分源码推理模块仍含 Python 3.10+ 类型标注；正式重新打包前需要消除该兼容性问题。
- H5/AEDAT4 属兼容输入；新格式或非标准字段布局应先做打开和时间轴验证。
