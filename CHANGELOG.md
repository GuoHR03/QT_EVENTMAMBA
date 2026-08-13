# Changelog

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

- 运行仍需要 NVIDIA GPU、ONNX Runtime GPU 及匹配的 CUDA/cuDNN 运行库。
- 当前开发环境的 ONNX Runtime 与自定义算子 CUDA 版本仍需在正式安装包前统一。
- 本版本先保留 WSL 兼容源码；后续完成干净 Windows 机器回归后再移除。
