# EventMamba Windows 原生推理可行性验证

## 当前结论

中心点和椭圆模型均已完成 Windows 原生 ONNX Runtime GPU 验证，并成功用同一个自定义算子 DLL 替换各自的 6 个 selective-scan `Loop`，同时把三级最远点采样从 Python 移入 ONNX 图中的 CPU 原生算子。当前真实 RAW 样本上，center 的 FPS+GPU 端到端 P50 约 `16.45 ms`，ellipse 约 `16.36 ms`。

因此，中心点和椭圆推理均不再需要用户侧 WSL、PyTorch、`mamba-ssm` 或 Python FPS 循环。两种模式都已经接入 Qt 的 Windows 后端，原生模型、椭圆矩阵、Custom Op DLL 和 CUDA 运行库也已进入安装包构建链；仍需继续补测其他 NVIDIA 显卡架构。

## 验证链路

### 1. PyTorch 等价实现与 ONNX 导出

原模型使用 `mamba_ssm` 自定义 CUDA 算子，不能直接完整导出。验证代码使用纯 PyTorch 双向 selective scan 作为导出路径：

- 单个 Mamba 组件最大绝对误差：`8.94e-08`
- 完整三层中心点模型最大绝对误差：约 `6.54e-06`
- 均通过 `allclose(rtol=1e-3, atol=1e-4)`

最初导出时，最远点采样 FPS 暂时作为三个显式输入：

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

与未融合的 Windows GPU 路径相比，当前实测加速约 360 倍；与 Windows CPU 路径相比，约快 98 倍。该历史结果包含一次完整 `session.run()`，但不包含 RAW 文件解析和 NumPy FPS 的时间。当前原生 FPS 模型的 `session.run()` 已包含三级 FPS。

### 5. 原生三级 FPS

`tools/onnx_insert_hierarchical_fps.py` 在保留原图其余节点的前提下，将
`fps0/fps1/fps2` 三个图输入替换为：

```text
events [B,3,1024] + fps_starts [B,3]
  -> com.eventmamba::HierarchicalFarthestPointSampling
  -> fps0 [B,512], fps1 [B,256], fps2 [B,128]
```

算子使用 C++/CPU 实现，保持 float32 距离、逐层升序重排、相对索引和最低
索引 tie-break。随机 batch=2、重复点和真实窗口的三级索引均与 NumPy oracle
逐项一致；非法 shape、起点及非有限输入会被拒绝。

- 独立算子：P50 `1.00 ms`，P95 `1.21 ms`
- center 旧 Python FPS+GPU：P50 `26.71 ms`
- center 原生 FPS+GPU：P50 `16.45 ms`，下降约 `38%`
- ellipse 旧 Python FPS+GPU：P50 `25.90 ms`
- ellipse 原生 FPS+GPU：P50 `16.36 ms`，下降约 `37%`
- 严格 ZMQ center 请求：P50 `33.46 -> 17.34 ms`，下降约 `48%`
- 严格 ZMQ ellipse 请求：P50 `33.82 -> 17.92 ms`，下降约 `47%`
- center/ellipse 的真实样本和 3 组合成样本均通过
  `allclose(rtol=1e-3, atol=1e-4)`

## 构建与验证工具

- `tools/build_selective_scan_ort.ps1`
- `tools/onnx_selective_scan_custom_op_probe.py`
- `tools/onnx_hierarchical_fps_custom_op_probe.py`
- `tools/onnx_insert_hierarchical_fps.py`
- `tools/onnx_native_fps_equivalence_probe.py`
- `tools/validate_windows_inference_artifacts.py`
- `tools/onnx_replace_selective_scan_loops.py`
- `tools/onnx_windows_runtime_probe.py`
- `tools/onnx_cuda_runtime.py`
- `tools/install_onnx_cuda_runtime.ps1`
- `tools/onnx_precomputed_fps_probe.py`
- `tools/onnx_fix_topk_k.py`
- `tools/extract_raw_inference_sample.py`
- `tools/onnx_saved_input_reference_probe.py`
- `tools/onnx_export_ellipse.py`
- `tools/onnx_exportable_eventmamba.py`
- `tools/onnx_ellipse_windows_probe.py`

### 6. 椭圆模型 Windows 结果

椭圆模型使用 `checkpoint/v14_new/P3best_checkpoint.pth` 和 `matrix_A.pt`，导出为 1024 维 VSA 输出的 ONNX 模型。`matrix_A` 单独转换为 NumPy 文件，运行时不需要 PyTorch。

- Windows CUDA 原始 1024 维输出最大绝对误差：`5.52e-05`
- 五参数解码最大绝对误差：`4.13e-05`
- PyTorch 与 NumPy VSA 解码最大绝对误差：`1.19e-07`
- `allclose(rtol=1e-3, atol=1e-4)`：通过
- 纯 ONNX GPU 10 次平均：`16.58 ms`
- 历史 Qt、ZMQ、Python FPS 与 GPU 完整链路预热后平均：`46.84 ms`
- 输出顺序：`[x, y, a, b, angle]`

正式 native-FPS ONNX、椭圆矩阵和 Custom Op DLL 作为版本化发布资产保留；其余实验模型、NPZ、构建目录、依赖头文件和运行库缓存继续忽略。

## 后续工作

1. 将完整 8 输入 `SelectiveScan` 节点直接写入导出图，避免目前保留的 `delta_a`、`delta_b_u` 中间大张量。
2. 发布时统一 CUDA Toolkit/运行库版本，补测其他 NVIDIA 架构和无 NVIDIA GPU 的 CPU 回退策略。
3. 在干净目标机继续执行 center/ellipse 打包后端长时间与模式切换回归。
4. 评估将原生 FPS 迁移到 CUDA 是否能抵消新增的 CPU/CUDA Memcpy；当前 CPU 版本已达到实时门槛，迁移前需证明净收益。
