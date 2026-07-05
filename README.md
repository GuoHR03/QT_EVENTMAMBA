# UI_Event

基于 PyQt6 的事件相机可视化与 EventMamba 推理工具，支持实时相机采集、离线文件回放、ROI 设置、去噪配置、WSL 后端推理，以及预测结果叠加显示。

## 当前状态

- 支持 `center` 中心点预测，输出 `(x, y)`
- 支持 `ellipse` 椭圆预测，输出 `[x, y, a, b, angle]`
- 支持 `RAW / HDF5 / H5 / AEDAT4` 离线回放
- `RAW / H5` 使用 Metavision `PeriodicFrameGenerationAlgorithm` 生成事件帧
- `AEDAT4` 使用项目内置 `EventFrameRenderer`，颜色与 Metavision SDK 调色板保持一致
- 支持 Activity / Trail / STC / AntiFlicker 等 Metavision 去噪配置
- 支持播放倍速、Palette、FPS 在播放过程中热更新，避免重新打开文件
- 支持 ROI 裁剪、推理模式切换、预测结果按帧时间戳叠加显示
- 前端与后端通过 ZeroMQ 通信
- Windows 端负责 UI、相机与显示，WSL 端负责模型推理

## 功能介绍

### 输入源与回放

- 实时 Metavision 相机：直接从设备读取事件流，支持硬件 ROI 尝试配置。
- `.raw` 文件：使用 Metavision `EventsIterator` 读取事件，并按事件时间戳进行真实时间回放。
- `.h5/.hdf5` 文件：自动识别事件字段，按统一事件格式转换后进行回放和推理。
- `.aedat4` 文件：使用 `dv_processing` 读取事件批次，再转换为统一事件格式进行显示、去噪和推理。

### 可视化

- 统一支持 `Dark / Light / CoolWarm / Gray` 调色板。
- `RAW / H5` 通过 Metavision 帧生成算法显示，`AEDAT4` 通过项目内置渲染器显示。
- 播放过程中可以直接调整 Palette、FPS 和播放倍速，不需要重启相机或重新打开离线文件。
- FPS 会影响显示帧切分和帧生成节奏，播放倍速只影响文件回放速度。

### 去噪与 ROI

- 去噪算法支持 `None / Activity / Trail / STC / AntiFlicker`。
- 去噪作用在显示和推理之前，便于对比过滤前后的可视化效果和推理稳定性。
- ROI 可用于限制显示、推理和归一化区域；实时 Metavision 设备会优先尝试硬件 ROI。

### 推理与叠加显示

- 支持 `center` 与 `ellipse` 两种预测模式。
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
│   ├── form.ui
│   └── choose_form.ui
├── backend/
│   ├── api.py                 # 后端统一门面
│   ├── camera_service.py      # 相机与录制管理
│   ├── inference_service.py   # WSL 推理服务管理
│   ├── Camera.py              # 底层相机/离线流处理
│   ├── camera_source_factory.py # 输入源创建与渲染器创建
│   ├── camera_source_runner.py # 不同输入源的运行分发
│   ├── event_frame_renderer.py # AEDAT4 事件帧渲染器
│   ├── h5_replay.py           # H5 文件回放
│   ├── h5_source.py           # H5 文件字段和分辨率解析
│   ├── metavision_source.py   # RAW/实时 Metavision 输入处理
│   ├── replay_clock.py        # 统一回放时钟
│   ├── replay_speed.py        # 播放倍速热更新控制
│   ├── NetworkThread.py       # ZMQ 请求响应线程
│   ├── realtime_inference.py  # 模型推理封装
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
