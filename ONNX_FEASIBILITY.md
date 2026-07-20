# EventMamba Windows 原生推理可行性验证

## 当前结论

中心点模型已经完成 Windows 原生 ONNX Runtime CPU/GPU 验证，并成功用自定义 CUDA 算子替换 6 个 selective-scan `Loop`。真实 RAW 样本的完整 GPU 推理从约 `6.65 s` 降至约 `18.4 ms`，输出仍与原 WSL/PyTorch 模型数值一致。

因此，发布版取消用户侧 WSL、PyTorch 和 `mamba-ssm` 依赖在技术上可行。尚未完成的是椭圆模型转换、Qt 后端接入、CUDA 运行库打包以及不同 NVIDIA 显卡的兼容测试。

## 验证链路

### 1. PyTorch 等价实现与 ONNX 导出

原模型使用 `mamba_ssm` 自定义 CUDA 算子，不能直接完整导出。验证代码使用纯 PyTorch 双向 selective scan 作为导出路径：

- 单个 Mamba 组件最大绝对误差：`8.94e-08`
- 完整三层中心点模型最大绝对误差：约 `6.54e-06`
- 均通过 `allclose(rtol=1e-3, atol=1e-4)`

最远点采样 FPS 从模型中移出，作为三个显式输入：

- `fps0`: `[1, 512]`
- `fps1`: `[1, 256]`
- `fps2`: `[1, 128]`

导出的中心点模型约 3.82 MB，包含三层双向 Mamba 对应的 6 个 ONNX `Loop`。

### 2. Windows CPU 与普通 CUDA EP

Windows 隔离环境使用 Python 3.13、ONNX Runtime GPU 1.27。修正 9 个旧导出器产生的动态 `TopK K` 参数后，模型可在 Windows 原生加载。

- Windows CPU，合成输入 10 次平均：约 `1.664 s`
- Windows CPU，真实 RAW 10 次平均：约 `1.807 s`
- 普通 ONNX CUDA EP，真实 RAW 10 次平均：约 `6.65 s`

普通 GPU 路径慢于 CPU 并非整体回退。性能剖析显示 6 个 `Loop` 产生约 55,026 个 CUDA 节点事件，大量极小 kernel 的逐时间步调度开销超过了计算收益。

### 3. Windows CUDA selective scan 自定义算子

新增的原生工程位于 `native/selective_scan_ort/`：

- Windows x64、FP32、仅前向推理
- 使用 ONNX Runtime 1.27 C API 获取 CUDA EP 的计算流
- 不依赖 PyTorch 或 `mamba-ssm`
- 当前模型状态维度为 16，POC 支持最大 32
- CUDA Toolkit 12.2 + VS2022 17.14 已在本机成功编译

三个实际层级的独立精度/性能结果：

| 输入形状 `[B,D,L,N]` | 最大绝对误差 | GPU 平均耗时 |
|---|---:|---:|
| `[1,128,512,16]` | `1.04e-07` | `1.47 ms` |
| `[1,256,256,16]` | `4.84e-08` | `1.02 ms` |
| `[1,512,128,16]` | `5.96e-08` | `0.69 ms` |

六个方向算子的独立调用耗时估算约 `6.36 ms`。

### 4. 完整模型 GPU 结果

`tools/onnx_replace_selective_scan_loops.py` 将完整模型的 6 个 `Loop + ConcatFromSequence` 替换为 6 个 `com.eventmamba::SelectiveScanCore` 节点。

真实样本来自 `record/recording_20260403_092545.raw` 的第一个 20 ms 事件窗口：

- 传感器分辨率：`1280x720`
- 模型输入事件点：`[1024,3]`
- WSL/PyTorch 参考输出：`[0.27019283, 0.47007358]`
- Windows 自定义 CUDA 输出：`[0.27025944, 0.47008979]`
- 相对参考最大绝对误差：约 `6.66e-05`
- `allclose(rtol=1e-3, atol=1e-4)`：通过
- 完整模型 10 次平均：`18.40 ms`
- 最快：`17.35 ms`
- 最慢：`21.02 ms`

与未融合的 Windows GPU 路径相比，当前实测加速约 360 倍；与 Windows CPU 路径相比，约快 98 倍。该结果包含一次完整 `session.run()`，但不包含 RAW 文件解析和 NumPy FPS 的时间。

## 构建与验证工具

- `tools/build_selective_scan_ort.ps1`
- `tools/onnx_selective_scan_custom_op_probe.py`
- `tools/onnx_replace_selective_scan_loops.py`
- `tools/onnx_windows_runtime_probe.py`
- `tools/onnx_cuda_runtime.py`
- `tools/install_onnx_cuda_runtime.ps1`
- `tools/onnx_precomputed_fps_probe.py`
- `tools/onnx_fix_topk_k.py`
- `tools/extract_raw_inference_sample.py`
- `tools/onnx_saved_input_reference_probe.py`

ONNX、NPZ、编译产物、依赖头文件和运行库缓存均位于已忽略目录，不会被误提交。

## 后续工作

1. 将完整 8 输入 `SelectiveScan` 节点直接写入导出图，避免目前保留的 `delta_a`、`delta_b_u` 中间大张量。
2. 将相同转换应用到椭圆模型，并验证 `matrix_A` 解码。
3. 把 FPS、ONNX Runtime 会话和自定义算子 DLL 接入 Qt 后端。
4. 发布时统一 CUDA Toolkit/运行库版本，补测其他 NVIDIA 架构和无 NVIDIA GPU 的 CPU 回退策略。
5. 完成 Windows 安装包后再移除正式产品中的 WSL 调用链。
