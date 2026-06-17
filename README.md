# UI_Event

基于 PyQt6 的事件相机可视化与 EventMamba 推理工具，支持实时相机采集、离线文件回放、ROI 设置、去噪配置、WSL 后端推理，以及预测结果叠加显示。

## 当前状态

- 支持 `center` 中心点预测，输出 `(x, y)`
- 支持 `ellipse` 椭圆预测，输出 `[x, y, a, b, angle]`
- 支持 `RAW / HDF5 / H5 / AEDAT4` 离线回放
- 支持 Activity / Trail / STC / AntiFlicker 等 Metavision 去噪配置
- 前端与后端通过 ZeroMQ 通信
- Windows 端负责 UI、相机与显示，WSL 端负责模型推理

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
│   ├── NetworkThread.py       # ZMQ 请求响应线程
│   ├── realtime_inference.py  # 模型推理封装
│   ├── protocol.py            # 后端消息协议
│   └── models/
├── libs/                      # Metavision 运行依赖
├── checkpoint/                # 模型权重目录
├── record/                    # 录制/离线文件目录
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
