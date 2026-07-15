# Version History

## v10

日期：2026-07-10

### Metavision PFG accumulation 对齐

- `PeriodicFrameGenerationAlgorithm` 创建时显式设置 `accumulation_time_us=frame_interval_us(fps)`。
- 30 FPS 时 accumulation time 约为 `33333us`，对应 Metavision Studio 常见的 `33.333ms` 配置。
- FPS 热更新会重建 PFG，因此 accumulation time 会随新的 FPS 一起更新。
- 对不支持构造参数 `accumulation_time_us` 的旧版 SDK，回退为创建后调用 accumulation setter。

### 三种输入源统一事件管线

- 新增 `EventPipeline`，RAW、H5、AEDAT4 共用事件格式转换、ROI、去噪和推理分流逻辑。
- 三种格式的推理事件统一按 `DEFAULT_NN_INTERVAL_MS=20ms` 跨读取批次切片，显示 FPS 不再决定推理窗口。
- 移除 RAW/AEDAT4 的重复推理切片和 H5 直接生成 payload 的独立路径。
- `InferencePayloadWorker` 只处理已经切好的事件窗口，每个窗口独立转换为一次模型输入。
- payload 构建抽为不依赖 Qt 的 `InferencePayloadProcessor`，便于测试和后续复用。
- 显示路径保持不变：RAW/H5 继续使用 Metavision PFG，AEDAT4 继续使用 `EventFrameRenderer`。
- 全量测试更新为 `182 passed`。

### 输入源接口与元数据统一

- 新增 `SourceMetadata`，统一提供输入类型、分辨率、起止时间、总时长和 seek 能力。
- 将输入源创建、Metavision Renderer 创建和格式元数据探测分别拆分到 `source_factory / renderer_factory / source_metadata`。
- 原 `camera_source_factory` 保留为兼容导出层，已有外部导入无需立即迁移。
- RAW 继续从 `.raw.tmp_index / *_info.json` 或扫描结果获取时长，H5 和 AEDAT4 继续直接读取文件内部信息，不生成额外 sidecar。
- RAW、H5、AEDAT4 分别实现统一 `EventSource` 的 `metadata / seek / run / close` 接口。
- `CameraThread` 改为通过统一工厂创建输入源，不再持有 `dv_reader / h5_file / mv_iterator` 等格式专属字段。
- `camera_source_runner` 不再判断文件格式，只负责建立 `EventPipeline`、运行 source 和统一关闭资源。

### SourceMetadataService 与线程注入

- RAW 时长 sidecar 读取、缓存、重复扫描抑制和后台扫描迁移到独立 `SourceMetadataService`。
- `CameraService` 只负责当前文件进度映射，不再持有 RAW 元数据实现和扫描线程状态。
- `CameraThread` 改为延迟导入并通过 `thread_factory` 注入，服务层不再因模块导入绑定 Qt DLL。
- 新增无 Qt 环境下的 `CameraService` 和元数据服务测试。

### PlaybackSession 与协作式停止

- 新增不依赖 Qt 的 `PlaybackSession`，统一管理 source、去噪初始化、事件管线和推理 worker 生命周期。
- 新增可独立测试的 `PlaybackCoordinator`，统一组装 source、推理 worker、录制、运行配置和进度回调。
- `CameraThread` 缩减为 Qt 线程与信号适配器，不再创建数据源或编排播放组件。
- `InferencePayloadWorker` 改为阻塞读取队列，空闲时不再每 `1ms` 轮询唤醒。
- 停止播放时通过运行标志和 STOP 哨兵协作退出，正常路径会等待 worker 完成并统一关闭 source。
- `QThread.terminate()` 仅保留为协作停止超过 3 秒后的最后兜底，并增加明确日志。

### 自然结束与主动停止语义

- 文件自然播放结束时，推理 worker 会先处理完已进入队列的事件窗口，再消费 STOP 哨兵退出。
- 用户停止或 seek 时会清空旧推理窗口并立即发送 STOP，避免旧时间点预测污染新播放位置。
- `EventSource` 新增统一 `request_stop()`，AEDAT4 和 Metavision 适配器会向底层 reader/iterator 转发中断请求。
- RAW 动态回放等待支持停止事件唤醒，停止时不再必须等待当前 replay sleep 完成。

### Renderer 接口与 AEDAT4 首帧对齐

- 新增统一 `FrameRenderer` 接口，Metavision PFG 包装器和 AEDAT4 `EventFrameRenderer` 均实现 `process_events / set_display_settings / reset / close`。
- `EventSource` 和 `EventPipeline` 内部统一使用 `renderer` 命名，不再把 AEDAT4 renderer 当作 frame generator。
- source 关闭时会同步关闭 renderer，避免旧 PFG 或图像回调被已结束会话继续持有。
- 移除 AEDAT4 首批事件整包立即显示的特殊路径，第一帧也从首事件时间开始按 `1/fps` 严格切分。
- 30 FPS 下 AEDAT4 第一帧累计窗口约为 `33.333ms`，边界后的事件会进入下一帧。

### Predictor 注册与网络结构解耦

- 新增通用 `PredictorSpec / PredictorRegistry`，通过模式注册具体网络实现，移除主推理入口中的模型类型条件分支。
- Center 与 Ellipse 的模型创建、权重加载、预热和输出解码迁移到独立 `eventmamba_predictors` 模块。
- `EventMambaPredictor` 只负责权重映射、模式切换和请求处理，并支持注入自定义 registry 与 `weights_by_mode`。
- 保留原 `realtime_inference` 中具体 Predictor 名称的延迟兼容导出。

### 主界面视觉与状态反馈

- 主画面新增输入类型、相机运行、模型加载和推理模式状态条。
- 日志区新增清空与折叠操作，收起后释放更多事件画面空间。
- 右侧控制区统一中文标签、宽度、间距和按钮主次层级。
- 录制、运行、加载中等状态改为动态属性驱动主题，不再在业务代码中写内联颜色。
- 日志输出改为只读并按信息、成功、警告和错误分色，保留原始日志文本与后端协议。
- 首帧到达后在输入状态中显示实际画面分辨率，模型加载期间使用等待光标和加载状态。
- 新增纯逻辑 UI 状态测试，避免依赖 Qt 图形环境。

### PlaybackConfig 与 ROI/去噪热更新

- 新增不可变 `PlaybackConfig` 和线程安全 `PlaybackConfigController`，统一 FPS、Palette、倍速、ROI、去噪和推理窗口参数。
- `AppSettings / BackendAPI / CameraService / CameraThread` 改为传递同一种配置快照，移除多份 last/current 参数状态。
- 离线 RAW/H5/AEDAT4 修改软件 ROI 时直接更新 `EventPipeline`，不再重建文件 source 或重置播放进度。
- 推理 payload 的 ROI 改为动态 getter，显示筛选和模型归一化始终读取同一个 ROI 快照。
- `NoiseFilterPipeline` 支持线程安全热更新；切换算法或阈值时原子重建 SDK 算法和 output buffer。
- 实时设备修改 ROI 时仍会重启 source 以应用硬件 ROI；单独修改去噪参数不重启。
- 推理窗口长度变化仍会重启 session，因为需要重建时间切片器。

## v9

日期：2026-07-10

### RAW 显示窗口与推理窗口解耦

- RAW/Metavision 输入源的读取 `delta_t` 改为按 UI FPS 计算的显示帧间隔，例如 30fps 对应约 33.333ms。
- 推理输入不再直接复用 RAW 显示事件包，而是在 RAW 事件循环中按 `DEFAULT_NN_INTERVAL_MS` 独立切片。
- 移除图像回调中的二次墙上时间节流，避免 Metavision 帧生成器已经按 FPS 输出后又被 UI 层额外丢帧。
- 该改动使 RAW 的显示参数更接近 Metavision Studio 的 `Frame rate / Accumulation time` 语义，同时保持推理模块继续使用 20ms 事件流。

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

### v8 备份点已知限制

- v8 备份点中，RAW 仍使用 `DEFAULT_NN_INTERVAL_MS` 作为 `EventsIterator(delta_t)`，因此推理事件窗口和 RAW 显示输入包仍存在耦合。
- v8 备份点中，UI 的 FPS 会传入 `PeriodicFrameGenerationAlgorithm`，但 RAW 读取分块默认仍为 20ms；这可能导致与 Metavision Studio 在相同 30fps 参数下存在显示差异。
- v9 已完成该优化：RAW 显示窗口按 `fps / accumulation time`，推理继续按 `DEFAULT_NN_INTERVAL_MS`。

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
