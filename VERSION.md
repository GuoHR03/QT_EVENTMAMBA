# Version History

## v8

日期：2026-07-10

### 播放进度条与拖拽回放

- 主界面新增离线文件播放进度条，显示当前播放时间和总时长。
- 支持拖拽进度条进行回放定位，拖拽释放后由后端重新按目标时间点启动文件回放。
- RAW 文件优先读取同目录 `.raw.tmp_index` 获取总时长；缺失时再回退到 `*_info.json` 或后台扫描。
- 修复 RAW 拖拽后进度条回到旧位置、拖拽无效果的问题。
- RAW seek 时会基于目标时间戳创建 `EventsIterator(start_ts=...)`，避免从文件开头重新等待。

### 回放参数热更新

- 播放倍速支持 `0.25x / 0.5x / 1x / 2x / 4x`，播放过程中修改不再重新打开文件。
- Palette 和 FPS 支持运行中热更新，RAW/H5 通过动态 Metavision frame generator 替换内部帧生成器。
- H5/AEDAT4 的帧切分会读取当前 FPS，后续帧按新的帧间隔继续生成。

### 可视化一致性与去噪

- RAW/H5/AEDAT4 统一支持 `Dark / Light / CoolWarm / Gray` 调色板。
- AEDAT4 内置 `EventFrameRenderer` 的颜色值按 Metavision SDK 调色板规则微调，降低与 Metavision 显示风格的差异。
- 去噪模块接入 AEDAT4 回放链路，RAW/H5/AEDAT4 均可在显示和推理前应用统一去噪。
- seek 过程中不再重复输出 `[NoiseFilter] Disabled` 初始日志，降低日志窗口噪声。

### 当前已知限制

- RAW 当前仍使用 `DEFAULT_NN_INTERVAL_MS` 作为 `EventsIterator(delta_t)`，因此推理事件窗口和 RAW 显示输入包仍存在耦合。
- UI 的 FPS 会传入 `PeriodicFrameGenerationAlgorithm`，但 RAW 读取分块默认仍为 20ms；这可能导致与 Metavision Studio 在相同 30fps 参数下存在显示差异。
- 下一步优化目标是将 RAW 显示窗口与推理窗口拆开：显示按 `fps / accumulation time`，推理继续按 `DEFAULT_NN_INTERVAL_MS`。

### 测试

- 新增 RAW 元数据读取、RAW seek、H5 seek、AEDAT4 seek、去噪日志抑制等单元测试。
- 当前测试覆盖提升到 `124` 项。

## v7

日期：2026-07-05

### 离线回放与可视化统一

- 统一支持 `RAW / H5 / HDF5 / AEDAT4` 离线事件文件回放。
- `RAW / H5` 使用 Metavision `PeriodicFrameGenerationAlgorithm` 生成事件帧。
- `AEDAT4` 新增项目内置 `EventFrameRenderer`，按 Metavision SDK 调色板颜色渲染事件帧。
- 新增 `backend/palettes.py` 中的 Metavision 调色板映射测试，保证 AEDAT4 与 Metavision 显示颜色一致。

### 回放时钟与播放控制

- 新增 `backend/replay_clock.py`，统一 H5 和 AEDAT4 的真实时间回放节奏。
- 新增 `backend/replay_speed.py`，支持播放倍速运行中热更新。
- RAW 文件回放新增动态 replay wrapper，播放中修改 `0.25x / 0.5x / 1x / 2x / 4x` 不再重新打开文件。
- 修正 Metavision `LiveReplayEventsIterator` 的参数语义差异，UI 中 `4x` 现在表示更快播放，`0.25x` 表示慢放。

### 显示设置热更新

- Palette 和 FPS 在播放过程中可以直接热更新，不再触发相机或文件重启。
- RAW/H5 通过动态 Metavision frame generator 代理替换内部渲染器。
- AEDAT4 renderer 支持运行中切换调色板。
- H5/AEDAT4 的帧切分会读取最新 FPS，调高或调低 FPS 会影响后续帧生成。

### 去噪、ROI 与推理链路

- AEDAT4 显示和推理链路接入统一 ROI 与去噪处理。
- 统一事件格式转换，减少不同文件格式在显示和推理前处理上的差异。
- 保持预测结果按事件时间戳匹配图像帧后叠加显示。

### 工具和测试

- 新增 AEDAT4 profiling 和 AEDAT4 转 H5 工具脚本。
- 新增 H5 source、AEDAT4 source、事件帧渲染器、回放时钟和动态 Metavision replay 的单元测试。
- 全量测试覆盖提升到 `112` 项。

## v6

日期：2026-06-17

### 结构优化

- 新增 `main.py` 作为统一启动入口
- 新增 `app/bootstrap.py`，集中处理运行路径、资源路径和 DLL 搜索路径
- 新增 `app/controller.py`，将 UI 与后端调用隔离
- 新增 `app/settings.py`，集中保存应用设置
- 新增 `app/view_state.py`，管理按钮和标签状态
- 新增 `app/prediction_state.py`，管理预测缓存与帧时间匹配
- 新增 `app/prediction_overlay.py`，管理预测结果解析和绘制
- 新增 `app/file_dialogs.py`、`app/paths.py`、`app/log_formatter.py`
- 新增 `backend/camera_service.py`，拆分相机和录制管理
- 新增 `backend/inference_service.py`，拆分 WSL 推理服务、ZMQ 线程和日志处理
- `backend/api.py` 简化为后端统一门面

### 清理

- 删除旧训练脚本 `train_ini30_vsa.py`
- 删除与训练脚本绑定的 `metrics.py`
- 主程序启动方式改为 `python main.py`

### 修复

- 修复 QtCreator 启动时可能找不到 `backend` 包的问题
- 修复 `NetworkThread.py` 中 `pickle` 导入异常导致推理响应解析失败的问题

## v5

日期：2026-05-10

- 接入 `ellipse` 椭圆预测链路
- 新增 `backend/models/eventmamba_v3.py`
- 新增 `backend/models/vsa.py`
- 椭圆模式支持读取权重同目录下的 `matrix_A.pt`
- 后端返回 `[x, y, a, b, angle]`
- 修复 ROI/模式窗口交互
- 改进模型加载和卸载时的 UI 状态
- 修复 WSL READY 握手和 ZMQ 超时重试逻辑

## v4

日期：2026-05-10

- 后端推理改为按当前模式加载单个权重
- 支持 `--center-weights` 和 `--ellipse-weights`
- 切换预测模式时自动重启推理后端
- 清理不再接入主流程的训练/实验脚本
- 模型文件整理到 `backend/models/`

## v3

日期：2026-05-07

- 新增 ROI 动态更新
- 新增 `center / ellipse` 预测模式切换
- 增强预测绘制逻辑
- 修复部分图像显示处理

## v2

日期：2026-04-20

- 修复 RAW 离线回放速度
- 增强预测结果与图像时间同步
- 改进 ZMQ 超时与异常处理
- 改进模型加载失败保护
- 改进 WSL 配置方式

## v1

日期：2026-04-20

- 事件相机可视化
- 基础数据处理
- EventMamba 推理接入
- PyQt6 图形界面
- 录制功能
