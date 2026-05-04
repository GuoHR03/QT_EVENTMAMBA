# UI_Event - 事件相机可视化与分析系统

基于 Qt6 的事件相机（Event Camera）可视化与分析系统，支持实时采集、离线回放、神经网络推理和预测结果可视化。

## 项目地址

https://github.com/GuoHR03/QT_EVENTMAMBA

## 核心功能

### 1. 多格式数据支持
- 事件相机实时采集（Metavision SDK）
- 离线文件回放（AEDAT4、RAW、H5/HDF5 格式）
- 录制功能（.raw 格式保存）

### 2. 事件数据处理
- 下采样与归一化
- 可配置的兴趣区域（ROI）裁剪
- 时间轴缩放与归一化
- 支持 640x480 和 1280x720 分辨率

### 3. 神经网络推理
- EventMamba 模型集成（通过 WSL 部署）
- 实时目标点预测
- 支持自定义模型权重加载

### 4. 可视化界面
- 事件流实时渲染（多种配色方案）
- 预测结果圆形标记
- 文件选择与参数配置

## 技术栈

| 组件 | 技术 |
|-----|-----|
| 前端 | PyQt6 |
| 事件处理 | Metavision SDK |
| 神经网络 | PyTorch + EventMamba |
| 通信 | ZMQ (Windows ↔ WSL) |
| 语言 | Python 3.8+ |

## 目录结构

```
UI_Event/
├── app/                     # Qt UI 界面
│   ├── widget.py           # 主窗口逻辑
│   ├── form.ui             # 主界面 UI
│   ├── choose_windows.py   # ROI 窗口
│   └── choose_form.ui      # ROI 界面 UI
├── backend/                 # 核心后端逻辑
│   ├── Camera.py           # 事件相机线程管理
│   ├── api.py              # 后端 API
│   ├── realtime_inference.py  # 实时推理
│   ├── NetworkThread.py    # 网络通信线程
│   └── Eventmamba/         # 神经网络模型
│       └── models/         # EventMamba 模型定义
├── libs/                    # Metavision SDK 库文件
├── linux_backend.py         # WSL 推理服务
└── pyproject.toml          # 项目配置
```

## 快速开始

### 环境要求
- Windows 10/11
- Python 3.8+
- WSL2 (用于神经网络推理)
- Metavision SDK

### 安装依赖

```bash
pip install PyQt6 pyzmq torch numpy opencv-python
```

### 运行

1. 启动 WSL 推理服务：
```bash
wsl -d EventMamba_mini python linux_backend.py --weights /path/to/model.pth
```

2. 运行主程序：
```bash
python app/widget.py
```

## 使用说明

### 相机实时采集
1. 选择配色方案
2. 设置 FPS
3. 点击"启动相机"

### 离线文件回放
1. 点击"选择文件"加载 AEDAT4/H5/RAW 文件
2. 可选加载 EventMamba 权重
3. 点击"启动相机"开始回放

### ROI 设置
1. 点击"选择窗口"
2. 在弹出的窗口中框选感兴趣区域
3. 应用设置

## 版本历史

详见 [VERSION.md](VERSION.md)
