# UI_Event

基于 PyQt6 的事件相机可视化与 EventMamba 推理工具，支持实时相机采集、离线文件回放、ROI 设置、去噪配置、Windows 原生 ONNX/CUDA 推理，以及预测结果叠加显示。0.2.0 发布包使用 Windows 原生独立后端，不需要 WSL 或目标机 Python；WSL 后端仅作为源码兼容选项保留。

## 当前状态

- Windows 原生后端支持 `center` 中心点预测，输出 `(x, y)`
- Windows 原生后端支持 `ellipse` 椭圆预测，输出 `[x, y, a, b, angle]`
- 支持 `RAW / HDF5 / H5 / AEDAT4` 离线回放
- `RAW / H5` 使用 Metavision `PeriodicFrameGenerationAlgorithm` 生成事件帧
- `AEDAT4` 使用项目内置 `EventFrameRenderer`，颜色与 Metavision SDK 调色板保持一致
- 两类 renderer 实现统一 `FrameRenderer` 接口，支持显示设置、reset 和 close 生命周期
- 支持 Activity / Trail / STC / AntiFlicker 等 Metavision 去噪配置
- 支持播放倍速、Palette、FPS 在播放过程中热更新，避免重新打开文件
- 使用不可变 `PlaybackConfig` 统一 FPS、Palette、倍速、ROI、去噪和推理窗口配置
- 离线文件的软件 ROI 与去噪支持播放中热更新，不重建 reader、不重置进度
- 支持离线文件播放进度条、总时长显示和拖拽定位回放
- RAW 文件优先使用 `.raw.tmp_index` 辅助获取总时长，提升进度条可用性
- RAW / H5 / AEDAT4 共用 `EventPipeline` 完成事件格式转换、ROI、去噪和推理分流
- RAW / H5 / AEDAT4 通过统一 `EventSource` 接口提供分辨率、时间范围、seek、主动停止和资源关闭能力
- `PlaybackSession` 统一管理输入源、事件管线、推理 worker 和协作式停止
- `PlaybackCoordinator` 负责数据源、推理 worker、录制、运行配置和进度的组装，`CameraThread` 仅保留 Qt 线程与信号适配
- `SourceMetadataService` 独立管理 RAW 时长 sidecar、缓存和后台扫描，`CameraService` 不再持有元数据实现
- 文件自然结束时排空待处理推理窗口；主动停止或 seek 时立即丢弃旧窗口
- 支持 ROI 裁剪、推理模式切换、预测结果按帧时间戳叠加显示
- 主画面提供输入类型、运行状态、模型状态和推理模式状态条
- 日志区域支持清空与折叠；右侧设置面板默认隐藏，通过画面右下角的设置按钮按需展开或收起
- 设置面板将播放控制、显示设置和 ROI 合并为一个逻辑栏，将模型管理和预测模式合并为另一个逻辑栏，各栏内部保留独立分区
- 运行日志按信息、成功、警告和错误分色，输入状态会在首帧后显示实际分辨率
- 前端与后端通过 ZeroMQ 通信
- 默认由 Windows 端同时负责 UI、相机、显示和 ONNX/CUDA 模型推理
- Windows 安装版内置独立的 `backend_runtime/UI_Event_Backend.exe`、模型和推理运行库
- 源码运行时仍可通过环境变量显式切换到原有 WSL 推理后端

## 功能介绍

### 输入源与回放

- 实时 Metavision 相机：直接从设备读取事件流，支持硬件 ROI 尝试配置。
- `.raw` 文件：使用 Metavision `EventsIterator` 读取事件，并按事件时间戳进行真实时间回放。
- `.h5/.hdf5` 文件：自动识别事件字段，按统一事件格式转换后进行回放和推理。
- `.aedat4` 文件：使用 `dv_processing` 读取事件批次，再转换为统一事件格式进行显示、去噪和推理。
- 三种格式对外返回统一 `SourceMetadata`；RAW 可从 sidecar 获取时长，H5/AEDAT4 直接读取文件内部元数据。
- 离线回放支持进度条显示当前时间/总时长，并可拖拽到目标位置继续播放。

### 可视化

- 统一支持 `Dark / Light / CoolWarm / Gray` 调色板。
- `RAW / H5` 通过 Metavision 帧生成算法显示，`AEDAT4` 通过项目内置渲染器显示。
- 播放过程中可以直接调整 Palette、FPS 和播放倍速，不需要重启相机或重新打开离线文件。
- FPS 会影响显示帧切分和帧生成节奏，播放倍速只影响文件回放速度。
- RAW 路径中，显示输入包按 `fps` 对应的帧间隔读取，`PeriodicFrameGenerationAlgorithm` 的 `accumulation_time_us` 也同步为该帧间隔。
- AEDAT4 从第一帧开始严格按 `1/fps` 切分事件窗口，30 FPS 时首帧同样为约 `33.333ms`。
- RAW / H5 / AEDAT4 的推理事件统一按 `DEFAULT_NN_INTERVAL_MS` 独立切片，不再复用显示帧窗口。

### 去噪与 ROI

- 去噪算法支持 `None / Activity / Trail / STC / AntiFlicker`。
- 去噪作用在显示和推理之前，便于对比过滤前后的可视化效果和推理稳定性。
- ROI 可用于限制显示、推理和归一化区域；实时 Metavision 设备会优先尝试硬件 ROI。
- 离线 RAW/H5/AEDAT4 的 ROI 和去噪参数可直接热更新；实时设备修改硬件 ROI 时仍会重启采集源。

### 推理与叠加显示

- Windows 原生 ONNX/CUDA 后端支持 `center` 与 `ellipse` 两种模式，并可在界面中切换。
- 椭圆网络输出 1024 维 VSA 向量，Windows 后端使用 NumPy 和导出的 `matrix_A.npy` 解码为五个椭圆参数。
- 设置 `EVENTMAMBA_INFERENCE_RUNTIME=wsl` 后仍可使用原有 PyTorch 模型作为兼容或对照后端。
- 网络实现通过 `PredictorRegistry` 按模式注册；模型加载、输入推理和输出解码封装在独立 Predictor 中。
- 更换同输入输出的网络时，只需实现 Predictor 并更新注册项，不需要修改事件读取、20ms 切片或请求处理主流程。
- Windows UI 通过 ZeroMQ 调用独立推理进程；源码模式使用独立 Python 环境，安装版使用冻结后的 `UI_Event_Backend.exe`。该进程通过 ONNX Runtime CUDA 和自定义 `selective_scan` CUDA 算子完成 EventMamba 推理。
- 推理结果携带事件时间戳，UI 按时间匹配图像帧后叠加绘制，降低结果和画面错位。

## 项目结构

```text
UI_Event/
├── main.py
├── linux_backend.py
├── windows_backend.py
├── UI_Event.spec             # UI onedir 打包配置
├── UI_Event_Backend.spec     # Windows 推理后端 onedir 打包配置
├── installer.iss             # Inno Setup 安装器配置
├── app/
│   ├── widget.py              # 主窗口协调
│   ├── controller.py          # UI 到后端的控制层
│   ├── view_state.py          # 按钮和标签状态
│   ├── prediction_state.py    # 预测缓存与时间匹配
│   ├── prediction_overlay.py  # 预测结果绘制
│   ├── file_dialogs.py        # 文件选择
│   ├── paths.py               # 默认路径
│   ├── settings.py            # 应用设置
│   ├── theme.py               # 样式
│   ├── ui_log.py              # 日志级别识别
│   ├── ui_status.py           # 输入源状态显示
│   ├── form.ui
│   └── choose_form.ui
├── backend/
│   ├── api.py                 # 后端统一门面
│   ├── camera_service.py      # 相机线程生命周期与配置
│   ├── inference_service.py   # Windows/WSL 推理服务管理
│   ├── windows_process.py     # Windows 后端进程生命周期
│   ├── windows_onnx_runtime.py # ONNX Runtime CUDA DLL 准备
│   ├── windows_onnx_predictor.py # Windows ONNX 中心点/椭圆推理
│   ├── ellipse_decoder.py     # 不依赖 PyTorch 的 VSA 椭圆解码
│   ├── Camera.py              # 底层相机/离线流处理
│   ├── camera_source_factory.py # 旧工厂导入兼容层
│   ├── camera_source_runner.py # 不同输入源的运行分发
│   ├── event_frame_renderer.py # AEDAT4 事件帧渲染器
│   ├── event_pipeline.py       # 统一事件预处理、显示与推理分流
│   ├── event_source.py         # 统一输入源接口与元数据
│   ├── frame_renderer.py       # 统一显示 renderer 接口
│   ├── h5_replay.py           # H5 文件回放
│   ├── h5_source.py           # H5 文件字段和分辨率解析
│   ├── inference_payload.py    # 推理事件窗口到模型输入的转换
│   ├── inference_worker_control.py # 推理队列结束语义
│   ├── metavision_source.py   # RAW/实时 Metavision 输入处理
│   ├── playback_session.py    # 播放会话与资源生命周期
│   ├── playback_config.py     # 不可变播放配置与线程安全快照
│   ├── playback_coordinator.py # 播放组件组装与运行协调
│   ├── replay_clock.py        # 统一回放时钟
│   ├── replay_speed.py        # 播放倍速热更新控制
│   ├── renderer_factory.py    # Metavision Renderer 与 PFG 创建
│   ├── source_factory.py      # RAW/H5/AEDAT4 输入源创建
│   ├── source_metadata.py     # 输入类型与格式元数据探测
│   ├── source_metadata_service.py # RAW 时长缓存与后台解析
│   ├── NetworkThread.py       # ZMQ 请求响应线程
│   ├── realtime_inference.py  # 注册式模型推理入口
│   ├── eventmamba_predictors.py # EventMamba 加载、推理与解码
│   ├── predictor_registry.py  # Predictor 规格与模式注册表
│   ├── protocol.py            # 后端消息协议
│   └── models/
├── libs/                      # Metavision 运行依赖
├── checkpoint/                # 模型权重目录
├── artifacts/                 # 转换后的 ONNX 模型（本机产物）
├── native/selective_scan_ort/ # Windows selective_scan 自定义算子
├── record/                    # 录制/离线文件目录
├── scripts/                   # 安装、运行和打包辅助脚本
├── tools/                     # 离线调试/实验工具脚本
└── wsl/                       # WSL 相关资源
```

## 运行方式

### 源码运行

建议从项目根目录启动：

```bash
python main.py
```

QtCreator 中也建议将启动脚本设置为 `main.py`。

### Windows 发布版

构建完整便携目录：

```powershell
.\scripts\build_installer.ps1 -Clean -SkipInstaller
```

启动文件为 `dist/UI_Event/UI_Event.exe`。必须整体保留 `dist/UI_Event/`，不能单独复制 EXE。

构建安装程序：

```powershell
.\scripts\build_installer.ps1 -Clean
```

安装包输出为 `installer/UI_Event_Setup.exe`。完整目录结构、构建依赖和干净电脑验收项目见 [PACKAGING.md](PACKAGING.md)。

## 环境要求

源码开发环境：

- Windows 10/11
- UI Python 3.8+
- 独立的 Windows 推理 Python 环境（当前验证环境为 Python 3.13）
- PyQt6
- pyzmq
- numpy
- opencv-python
- ONNX Runtime GPU、CUDA 与 cuDNN 运行库
- Metavision SDK

0.2.0 安装版已携带 Python 冻结运行时、ONNX Runtime、构建时收集的 CUDA/cuDNN DLL、模型、自定义算子和 Metavision 用户态运行库。目标电脑不需要安装 Python、WSL、CUDA Toolkit、cuDNN 或完整 Metavision SDK，但仍需要兼容的 NVIDIA GPU/驱动、Microsoft Visual C++ 2015–2022 x64 Runtime，以及实时相机所需的 Metavision 设备驱动。

只有在源码模式使用兼容后端时才需要 WSL2、PyTorch、Mamba 及其 Linux 环境。默认安装器不会安装或修改 WSL。

## 环境变量

```text
EVENTMAMBA_INFERENCE_RUNTIME=windows
EVENTMAMBA_WINDOWS_PYTHON=.venv-onnx-win/Scripts/python.exe
EVENTMAMBA_WINDOWS_BACKEND_EXECUTABLE=backend_runtime/UI_Event_Backend.exe
EVENTMAMBA_CENTER_ONNX_MODEL=artifacts/eventmamba_center_selective_scan_cuda.onnx
EVENTMAMBA_ELLIPSE_ONNX_MODEL=artifacts/eventmamba_ellipse_selective_scan_cuda.onnx
EVENTMAMBA_ELLIPSE_MATRIX=artifacts/eventmamba_ellipse_matrix_A.npy
EVENTMAMBA_SELECTIVE_SCAN_DLL=native/selective_scan_ort/bin/eventmamba_selective_scan.dll
EVENTMAMBA_BACKEND_READY_TIMEOUT_S=180
METAVISION_SDK_PATH=E:\Metavision\Prophesee
```

Windows 是默认值，因此通常不需要设置第一项。`EVENTMAMBA_WINDOWS_PYTHON` 仅用于源码模式；冻结版会直接启动 `EVENTMAMBA_WINDOWS_BACKEND_EXECUTABLE` 指向的程序，其默认值为安装根目录中的 `backend_runtime/UI_Event_Backend.exe`。源码兼容的 WSL 后端可使用：

```text
EVENTMAMBA_INFERENCE_RUNTIME=wsl
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
```

## 模型文件

Windows 推理需要两个转换后的 ONNX 模型、椭圆解码矩阵和匹配的自定义 CUDA DLL。默认位置为：

```text
artifacts/eventmamba_center_selective_scan_cuda.onnx
artifacts/eventmamba_ellipse_selective_scan_cuda.onnx
artifacts/eventmamba_ellipse_matrix_A.npy
native/selective_scan_ort/bin/eventmamba_selective_scan.dll
```

0.2.0 构建脚本会把这四项复制到发布根目录中的相同相对路径，并在生成安装器前检查其存在性。WSL 源码兼容模式的 `ellipse` 仍需要权重文件同目录下存在 `matrix_A.pt`：

```text
checkpoint/
└── your_model/
    ├── P3best_checkpoint.pth
    └── matrix_A.pt
```

## 维护说明

- `build/`、`dist/`、`installer/` 是构建产物，可以重新生成
- `__pycache__/` 是 Python 缓存，可以删除
- `eventmamba_backend.log` 是运行日志，可以删除
- 安装版录像和日志写入 `%LOCALAPPDATA%\UI_Event`，卸载程序不会删除用户数据
- 主程序不再依赖训练脚本，训练/实验脚本不应混入主运行链路
- Windows 安装器不再携带或自动导入 WSL；手动 WSL 脚本仅供源码兼容流程使用
- 源码和文档统一使用 UTF-8 编码；如果 PowerShell 显示中文乱码，请用 `Get-Content -Encoding UTF8` 查看
- 当前 RAW 显示窗口与推理窗口已经拆分：显示按 `fps / accumulation time` 驱动，推理继续按 `DEFAULT_NN_INTERVAL_MS` 切片。
