# UI_Event

基于 `PyQt6` 的事件相机可视化与推理工具，支持实时采集、离线回放、ROI 设置、WSL 后端推理，以及预测结果叠加显示。

项目地址：
`https://github.com/GuoHR03/QT_EVENTMAMBA`

## 当前状态

- `center` 模式已接入真实网络，输出瞳孔中心点 `(x, y)`
- `ellipse` 模式已接入真实推理链路，输出 `[x, y, a, b, angle]`
- 前后端按当前模式单次只加载一个权重文件
- 椭圆模式会自动读取权重同目录下的 `matrix_A.pt`

## 主要功能

- 事件相机实时采集
- `AEDAT4 / RAW / H5(HDF5)` 离线回放
- ROI 选择与动态更新
- `center / ellipse` 两种预测模式切换
- Windows -> WSL 的 `ZeroMQ` 推理通信
- 推理结果与图像时间对齐显示
- 椭圆预测结果前端叠加绘制

## 目录结构

```text
UI_Event/
├── app/
│   ├── widget.py
│   ├── form.ui
│   ├── choose_windows.py
│   └── choose_form.ui
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── Camera.py
│   ├── NetworkThread.py
│   ├── realtime_inference.py
│   └── models/
│       ├── __init__.py
│       ├── eventmamba_v1.py
│       ├── eventmamba_v3.py
│       ├── mamba_layer.py
│       ├── modules.py
│       └── vsa.py
├── linux_backend.py
├── train_ini30_vsa.py
├── VERSION.md
└── pyproject.toml
```

## 推理模式说明

### `center`

- 前端启动后端时传入 `--center-weights <path>`
- 后端加载中心点网络
- 输出格式为 `[x, y]`

### `ellipse`

- 前端启动后端时传入 `--ellipse-weights <path>`
- 后端加载椭圆网络 `eventmamba_v3`
- 同时自动读取权重同目录下的 `matrix_A.pt`
- 输出格式为 `[x, y, a, b, angle]`
- 其中：
  - `x, y, a, b` 归一化到 `[0, 1]`
  - `angle` 范围为 `[-0.5π, 0.5π]`

## 椭圆绘制说明

- 椭圆绘制在前端 [app/widget.py](/E:/Code/Qt/UI_Event-main/app/widget.py) 中完成
- `angle` 会先从弧度转换为角度，再交给 `QPainter.rotate()`
- `a / b` 会按当前图像尺寸缩放
- 如果结果来自裁剪后的 ROI，椭圆半轴会按 ROI 尺寸缩放

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
wsl -d EventMamba_mini /opt/miniconda3/envs/eventmamba/bin/python linux_backend.py --center-weights /path/to/center_model.pth
```

`ellipse` 模式：

```bash
wsl -d EventMamba_mini /opt/miniconda3/envs/eventmamba/bin/python linux_backend.py --ellipse-weights /path/to/ellipse_model.pth
```

## 椭圆模型文件要求

椭圆模式至少需要：

- 椭圆权重文件，例如 `P3best_checkpoint.pth`
- 同目录下的 `matrix_A.pt`

示例：

```text
some_folder/
├── P3best_checkpoint.pth
└── matrix_A.pt
```

## 环境变量

```bash
EVENTMAMBA_WSL_DISTRO=EventMamba_mini
EVENTMAMBA_LINUX_PYTHON=/opt/miniconda3/envs/eventmamba/bin/python
```

## 使用说明

1. 启动程序后选择输入源或实时相机
2. 根据需要打开 ROI 窗口，并切换 `center / ellipse`
3. 选择与当前模式对应的权重文件
4. 点击加载模型
5. 启动相机或开始离线回放
6. 在主界面查看中心点或椭圆预测结果

## 版本记录

详见 [VERSION.md](VERSION.md)
