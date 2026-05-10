# UI_Event

基于 PyQt6 的事件相机可视化与推理工具，支持实时采集、离线回放、ROI 设置、WSL 后端推理，以及预测结果叠加显示。

项目地址：

`https://github.com/GuoHR03/QT_EVENTMAMBA`

## 当前状态

- `center` 模式已接入实际网络，输出瞳孔中心点 `(x, y)`
- `ellipse` 模式已预留独立网络接口，但尚未接入实际椭圆模型
- 前后端已改为“单次只传一个权重文件”，并根据当前模式加载到对应位置

## 主要功能

- 事件相机实时采集
- AEDAT4 / RAW / H5(HDF5) 离线回放
- ROI 选择与动态更新
- Windows -> WSL 的 ZMQ 推理通信
- 推理结果与图像时间对齐显示
- `center / ellipse` 两种预测模式切换

## 目录结构

```text
UI_Event/
├─ app/
│  ├─ widget.py
│  ├─ form.ui
│  ├─ choose_windows.py
│  └─ choose_form.ui
├─ backend/
│  ├─ __init__.py
│  ├─ api.py
│  ├─ Camera.py
│  ├─ NetworkThread.py
│  ├─ realtime_inference.py
│  └─ models/
│     ├─ __init__.py
│     ├─ eventmamba_v1.py
│     ├─ mamba_layer.py
│     └─ modules.py
├─ libs/
├─ linux_backend.py
├─ VERSION.md
└─ pyproject.toml
```

## 推理模式说明

### center

- 前端启动后端时传入 `--center-weights <path>`
- 后端加载中心点网络
- 当前可正常推理

### ellipse

- 前端启动后端时传入 `--ellipse-weights <path>`
- 后端进入独立的椭圆接口
- 目前仅保留占位类，尚未接入真实椭圆网络

## 运行方式

### 环境要求

- Windows 10/11
- Python 3.8+
- WSL2
- Metavision SDK

### 安装依赖

```bash
pip install PyQt6 pyzmq torch numpy opencv-python
```

### 启动主程序

```bash
python app/widget.py
```

### 手动启动 WSL 后端

`center` 模式：

```bash
wsl -d EventMamba_mini python linux_backend.py --center-weights /path/to/center_model.pth
```

`ellipse` 模式：

```bash
wsl -d EventMamba_mini python linux_backend.py --ellipse-weights /path/to/ellipse_model.pth
```

## 环境变量

```bash
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
```

## 使用说明

1. 启动程序后选择输入源或实时相机
2. 设置 FPS、配色和 ROI
3. 根据模式选择对应权重文件
4. 加载模型后启动相机或回放
5. 在主界面查看推理结果

## 版本记录

详见 [VERSION.md](VERSION.md)
