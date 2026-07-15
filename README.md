# UI_Event

基于 PyQt6 的事件相机可视化与 EventMamba 推理工具，支持实时相机采集、离线文件回放、ROI 设置、去噪配置、WSL 后端推理，以及预测结果叠加显示。

## 当前状态

- 支持 `center` 中心点预测，输出 `(x, y)`
- 支持 `ellipse` 椭圆预测，输出 `[x, y, a, b, angle]`
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
- 日志区域支持清空与折叠，右侧显示和处理参数使用统一控件层级
- 运行日志按信息、成功、警告和错误分色，输入状态会在首帧后显示实际分辨率
- 前端与后端通过 ZeroMQ 通信
- Windows 端负责 UI、相机与显示，WSL 端负责模型推理

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

- 支持 `center` 与 `ellipse` 两种预测模式。
- 网络实现通过 `PredictorRegistry` 按模式注册；模型加载、输入推理和输出解码封装在独立 Predictor 中。
- 更换同输入输出的网络时，只需实现 Predictor 并更新注册项，不需要修改事件读取、20ms 切片或请求处理主流程。
- Windows UI 负责事件采集、可视化和请求发送，WSL 后端负责 EventMamba 模型推理。
- 推理结果携带事件时间戳，UI 按时间匹配图像帧后叠加绘制，降低结果和画面错位。

## 项目结构

```text
UI_Event/
├── main.py
├── linux_backend.py
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
│   ├── inference_service.py   # WSL 推理服务管理
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
├── record/                    # 录制/离线文件目录
├── scripts/                   # 安装、运行和打包辅助脚本
├── tools/                     # 离线调试/实验工具脚本
└── wsl/                       # WSL 相关资源
```

## 运行方式

建议从项目根目录启动：

```bash
python main.py
```

QtCreator 中也建议将启动脚本设置为 `main.py`。

## 环境要求

- Windows 10/11
- Python 3.8+
- PyQt6
- pyzmq
- numpy
- opencv-python
- torch
- WSL2
- Metavision SDK

## 环境变量

```bash
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
EVENTMAMBA_BACKEND_READY_TIMEOUT_S=180
METAVISION_SDK_PATH=E:\Metavision\Prophesee
```

## 模型文件

`ellipse` 模式需要权重文件同目录下存在 `matrix_A.pt`：

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
- 主程序不再依赖训练脚本，训练/实验脚本不应混入主运行链路
- 源码和文档统一使用 UTF-8 编码；如果 PowerShell 显示中文乱码，请用 `Get-Content -Encoding UTF8` 查看
- 当前 RAW 显示窗口与推理窗口已经拆分：显示按 `fps / accumulation time` 驱动，推理继续按 `DEFAULT_NN_INTERVAL_MS` 切片。
