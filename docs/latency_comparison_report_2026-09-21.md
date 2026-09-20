# EventMamba / RandLA 推理延迟对比报告

- 测试日期：2026-09-21
- 测试平台：Windows + WSL2（`EventMamba_mini`）
- GPU：NVIDIA GeForce RTX 3050 Ti Laptop GPU
- 输入：单批次事件张量，形状为 `(1024, 3)`，`float32`
- 任务：椭圆预测（VSA 输出解码为 5 个参数）
- 测试方法：每组预热 20 次，随后连续测量 100 次
- 往返口径：Windows 客户端完成 ZMQ 请求编码、发送、后端处理、响应接收和解码的总时间
- 推理口径：后端从进入预测器到完成模型推理、VSA 解码及结果回传 CPU 的时间

## 最终结果

| 模型 | 运行路径 | 推理平均 | 推理 P50 | 推理 P95 | 往返平均 | 往返 P50 | 往返 P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| EventMamba | WSL / PyTorch | 103.30 ms | 97.17 ms | 140.43 ms | 105.07 ms | 98.98 ms | 142.25 ms |
| EventMamba | Windows ONNX / CUDA | 16.93 ms | 16.88 ms | 17.29 ms | 18.18 ms | 18.14 ms | 18.51 ms |
| RandLA | WSL / PyTorch | 22.27 ms | 18.04 ms | 32.75 ms | 66.50 ms | 61.36 ms | 77.78 ms |
| RandLA | Windows ONNX / CUDA | 16.53 ms | 16.52 ms | 17.03 ms | 17.70 ms | 17.69 ms | 18.34 ms |

## 路径收益

| 模型 | WSL 往返平均 | ONNX 往返平均 | 平均节省 | ONNX 加速比 |
|---|---:|---:|---:|---:|
| EventMamba | 105.07 ms | 18.18 ms | 86.89 ms | 5.78× |
| RandLA | 66.50 ms | 17.70 ms | 48.80 ms | 3.76× |

| 模型 | WSL 推理平均 | ONNX 推理平均 | 推理耗时降低 |
|---|---:|---:|---:|
| EventMamba | 103.30 ms | 16.93 ms | 83.6% |
| RandLA | 22.27 ms | 16.53 ms | 25.7% |

## 结果解释

EventMamba 的 WSL/PyTorch 实现使用三级最远点采样（FPS），分别执行 512、256 和 128 次相互依赖的迭代，总计 896 次。该实现会产生大量串行的小型 CUDA kernel 调度，因此模型推理平均达到 103.30 ms。

EventMamba ONNX 模型使用原生 CUDA `HierarchicalFarthestPointSampling` 自定义算子替代 PyTorch FPS 循环，并使用 CUDA `SelectiveScanCore` 执行 Mamba selective-scan，因此推理降低到约 17 ms。

RandLA 的 WSL/PyTorch 实现采用随机采样，避免了串行 FPS，因此模型本体明显快于 WSL EventMamba。不过本次 Windows 客户端到 WSL RandLA 后端的测量中，推理外开销平均约为 44.24 ms；同一协议在本地 Windows ONNX 后端中的推理外开销约为 1.17 ms。该额外时间属于 Windows/WSL 边界、ZMQ/TCP 和进程调度的组合开销，不应计入模型计算性能。

为排除模型影响，另使用项目的最小 ZMQ 延迟服务器测试 Windows 到 WSL 的 12,288 字节原始单帧请求，100 次平均为 43.99 ms、P95 为 44.24 ms；使用正式事件协议的双帧请求平均为 44.00 ms、P95 为 44.32 ms。这证明约 44 ms 来自当前 WSL NAT/localhost 转发链路，而不是 RandLA 模型或消息序列化。直接连接 WSL 虚拟网卡地址受到当前 Windows 防火墙策略阻止，未作为应用默认方案。

在 Windows ONNX/CUDA 路径中，两种模型的平均推理差异小于 0.5 ms，属于正常运行波动。当前可以认为它们处于同一延迟等级。

## 结论与建议

1. 正式 UI 运行默认使用 Windows ONNX/CUDA。
2. EventMamba ONNX 和 RandLA ONNX 的实际往返延迟均约为 18 ms，且 P95 稳定。
3. WSL/PyTorch 保留为新权重兼容性验证、模型调试和 ONNX 导出前的后备路径。
4. 比较模型结构时优先查看“推理时间”；评估 UI 体验时使用“往返时间”。
5. 延迟结果会随 GPU 功耗状态、后台负载、驱动版本和输入内容变化，发布前应在目标机器上重复测量。
6. 如果必须长期使用 WSL 实时推理，可进一步评估 WSL mirrored networking 或经过授权的虚拟网卡防火墙规则；这属于主机网络配置，不应由应用静默修改。

## 本次审查后的工程优化

- Windows 模型选择器的正式模型筛选已同时包含 `*_native_fps.onnx` 和 RandLA 的 `*_native.onnx`。
- 安装包构建会分别启动 EventMamba 与 RandLA 椭圆模型做端到端预测冒烟测试，避免只校验文件结构却没有真实执行 RandLA。
- 后端基准现在分别输出模型推理、完整请求往返和其他开销，后续可直接复现本报告的统计口径。
- native ONNX 模型统一在后端就绪前预热，并使用独立随机数生成器，不消耗正式推理采样序列。本机 EventMamba 第一笔正式请求由预热前的 625.07 ms 降至预热后的 19.57 ms；稳态表格数据不受影响。
- 尝试过将 WSL RandLA 的随机采样和 KNN 全量排序替换为 `randperm`/`topk`，但实测产生明显性能回归，因此未保留该改动。
- 已完成一次干净的便携发布构建；冻结后的 EventMamba 与 RandLA 后端均通过中心点、椭圆和模式切换预测测试。

## 对应资产

- EventMamba ONNX：`artifacts/eventmamba_ellipse_native_fps.onnx`
- EventMamba VSA 矩阵：`artifacts/eventmamba_ellipse_matrix_A.npy`
- RandLA ONNX：`artifacts/eventmamba_ellipse_randla_native.onnx`
- RandLA VSA 矩阵：`artifacts/eventmamba_ellipse_randla_matrix_A.npy`
- EventMamba WSL 权重：`checkpoint/v14_new/P3best_checkpoint.pth`
- RandLA WSL 权重：`checkpoint/vsa_dynamic_sampling_cached_fixedA_Gradient_Clipping_randla_randomsample/P3best_checkpoint.pth`
