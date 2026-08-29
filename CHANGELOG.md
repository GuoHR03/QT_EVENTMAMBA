# Changelog

正式发布使用语义版本号；开发过程中的 `v1`–`v10` 快照记录见
[VERSION.md](VERSION.md)，两者不是同一套版本编号。

## [Unreleased]

### Added

- Windows 原生推理服务统一为严格 JSON + float32 二进制 ZeroMQ 协议。
- 增加后端 PID/nonce 身份校验、请求代际隔离和更完整的推理生命周期状态。
- 增加原生三级最远点采样算子、native-FPS center/ellipse 模型和资产校验工具。
- 增加离线回放进度、Seek、倍速、显示参数热更新，以及更完整的 UI/协议/推理测试。

### Changed

- Windows 安装版改为 UI 与推理后端两个独立 PyInstaller onedir 程序，默认不再依赖 WSL。
- 发布包携带 ONNX Runtime、构建时收集的 CUDA/cuDNN DLL、正式模型、自定义算子和 Metavision 用户态运行库。
- 精简 Metavision Debug/CPython 3.9 二进制，并强化安装包资产与冻结后端冒烟检查。

### Validation

- 当前自动化测试：`384 passed`。
- native-FPS center/ellipse 模型、严格 ZMQ 请求和模式切换已加入专项验证工具。

### Known limitations

- 发布 UI 固定使用 CPython 3.8 ABI，但部分源码推理模块仍含 Python 3.10+ 类型标注，重新发布前需要修复。
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
