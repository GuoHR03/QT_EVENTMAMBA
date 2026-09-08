# Changelog

正式发布使用语义版本号；开发过程中的 `v1`–`v10` 快照记录见
[VERSION.md](VERSION.md)，两者不是同一套版本编号。

## [Unreleased]

## [0.2.1] - 2026-09-08

### Added

- Windows 原生推理服务统一为严格 JSON + float32 二进制 ZeroMQ 协议。
- 增加后端 PID/nonce 身份校验、请求代际隔离和更完整的推理生命周期状态。
- 增加原生三级最远点采样算子、native-FPS center/ellipse 模型和资产校验工具。
- 增加离线回放进度、Seek、倍速、显示参数热更新，以及更完整的 UI/协议/推理测试。
- 增加双 Python 运行时契约、分环境固定依赖清单和发布前版本校验。
- 增加 Windows GitHub Actions CI，覆盖完整测试、类型检查、构建脚本语法和 CPython 3.8 Metavision 运行时冒烟检查。
- 原生 ONNX Runtime算子构建脚本改为自动探测 Visual Studio、CMake、Ninja和CUDA，并支持显式路径覆盖及只解析模式。
- 相机和推理网络线程改为可重试的协作式停止，移除 `QThread.terminate()`；H5/AEDAT4回放等待可由停止事件主动唤醒。
- 相机、回放和推理服务统一使用线程安全的五态生命周期状态机，并增加非法转换、失败恢复和状态回调测试。
- 增加统一应用关闭协调器，串行关闭 Qt 资源与推理后端，合并重复关闭请求，并支持从失败阶段安全重试。
- 拆分过大的 `MainWindow`，将动态布局、事件视口、相机/回放交互和推理操作入口迁移到可独立测试的 UI 组件。
- 收窄 UI 组件依赖范围，以不可变端口对象替代对整个 `MainWindow` 的引用，并支持注入文件选择与事件循环回调。
- 增加独立 `.venv-dev` 测试环境契约、一键环境创建/完整检查脚本，以及真实 MainWindow 离屏构造和协作式关闭检查；Qt 测试桩不再静默启用。

### Changed

- Windows 安装版改为 UI 与推理后端两个独立 PyInstaller onedir 程序，默认不再依赖 WSL。
- 发布包携带 ONNX Runtime、构建时收集的 CUDA/cuDNN DLL、正式模型、自定义算子和 Metavision 用户态运行库。
- 精简 Metavision Debug/CPython 3.9 二进制，并强化安装包资产与冻结后端冒烟检查。

### Validation

- 完整自动化测试套件通过。
- native-FPS center/ellipse 模型、严格 ZMQ 请求和模式切换已加入专项验证工具。

### Known limitations

- 发布 UI 固定使用 CPython 3.8 ABI，推理源码和依赖必须由独立的 CPython 3.13 后端加载。
- 原生 DLL 当前构建目标为 CUDA 架构 7.5 和 8.6；其他 NVIDIA 架构仍需补测或重新编译。
- Windows CPU 推理回退尚未完成正式验证。

## [0.2.0] - 2026-07-21

### Added

- Windows 原生 ONNX Runtime CUDA 推理后端，不再要求中心点推理经过 WSL。
- Windows `selective_scan` CUDA 自定义算子，替换 ONNX 中的 6 个逐步 `Loop`。
- 椭圆模型 Windows ONNX 转换，输出 1024 维 VSA 后解码为 `[x, y, a, b, angle]`。
- 不依赖 PyTorch 的 NumPy `matrix_A` 椭圆解码器。
- 中心点与椭圆 ONNX 模型、椭圆矩阵和已验证的自定义算子 DLL。
- 椭圆模型导出、CUDA 对比和完整 Qt 链路验证工具。

### Changed

- 默认推理运行时从 WSL 切换为 `Windows ONNX CUDA`。
- Qt 模型选择器改为选择 ONNX 模型，并支持中心点/椭圆模式切换。
- 后端启动、健康检查和网络错误提示不再绑定 WSL 表述。
- WSL 后端暂时保留为显式兼容选项，可通过 `EVENTMAMBA_INFERENCE_RUNTIME=wsl` 启用。

### Validation

- 中心点真实 RAW 样本纯模型 GPU 推理约 `18.4 ms`。
- 椭圆纯模型 GPU 推理 10 次平均约 `16.58 ms`。
- 椭圆 Qt、ZMQ、FPS 和 GPU 完整链路预热后平均约 `46.84 ms`。
- 椭圆五参数相对 PyTorch 导出参考最大绝对误差约 `4.13e-05`。
- 自动化测试：`190 passed`。

### Packaging notes

- 这是 `0.2.0` 发布时的历史状态：运行环境需要用户自行准备 NVIDIA GPU、ONNX Runtime GPU 及匹配的 CUDA/cuDNN 运行库。
- 当时 ONNX Runtime 与自定义算子 CUDA 版本尚待正式安装包统一；`Unreleased` 中的后续打包链已经将这些运行库收入安装版。
- WSL 兼容源码继续保留，但后续安装版已经默认使用 Windows 独立后端，不再安装或修改 WSL。
