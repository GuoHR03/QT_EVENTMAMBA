# Metavision 去噪算法说明

本文说明当前项目中接入的 Metavision SDK CV 去噪算法，以及主界面设置面板中
“去噪算法”和“阈值 (μs)”的调参方式。

操作路径：点击主画面右下角的设置按钮，展开“去噪”区域，选择算法和阈值，
最后点击“应用设置”。去噪初始化或运行失败时，
程序会在日志中报告原因，并继续传递未过滤的原始事件。

## 当前项目接入方式

当前项目在事件流进入显示和推理之前执行去噪。实时 Metavision、`.raw`、`.h5/.hdf5` 和 `.aedat4` 都会先转换为统一事件格式，再按当前 UI 去噪配置处理：

| 输入来源 | 显示画面 | 推理事件 |
| --- | --- | --- |
| 实时 Metavision 设备 | 使用滤波后事件 | 使用滤波后事件 |
| `.raw` 回放 | 使用滤波后事件 | 使用滤波后事件 |
| `.h5/.hdf5` 回放 | 使用滤波后事件 | 使用滤波后事件 |
| `.aedat4` 回放 | 使用滤波后事件 | 使用滤波后事件 |

UI 中“去噪算法”的可选项：

| UI 名称 | SDK 算法 |
| --- | --- |
| `None` | 不启用去噪 |
| `Activity` | `ActivityNoiseFilterAlgorithm` |
| `Trail` | `TrailFilterAlgorithm` |
| `STC` | `SpatioTemporalContrastAlgorithm` |
| `AntiFlicker` | `AntiFlickerAlgorithm` |

“阈值 (μs)”的单位是微秒。`10000μs` 等于 `10ms`。

不同算法对阈值方向的解释并不完全相同：

| 算法 | 阈值增大后的主要效果 |
| --- | --- |
| Activity | 允许更久以前的邻域活动作为依据，通常保留更多事件 |
| Trail | 同一像素需要间隔更久才再次通过，通常抑制更多拖尾 |
| STC | 允许更久以前的时空关联作为依据，通常保留更多事件 |
| AntiFlicker | 允许周期差异更大，频闪识别更宽容，但也更容易误伤周期性运动 |

建议每次只调整一个参数，并结合画面、事件率和推理稳定性判断效果。

## ActivityNoiseFilterAlgorithm

作用：过滤孤立事件。它只保留在当前事件附近、过去一段时间窗口内出现过相似活动的事件。

适合场景：

- 背景随机噪声明显。
- 场景中目标运动比较连续。
- 希望先用一个稳妥的通用去噪算法。

主要参数：

| 参数 | 当前 UI | 含义 |
| --- | --- | --- |
| `width` | 自动使用相机宽度 | 事件流宽度 |
| `height` | 自动使用相机高度 | 事件流高度 |
| `threshold` | “阈值 (μs)” | 活动时间窗口，单位微秒 |

调参建议：

- 噪声仍然很多：适当减小阈值。
- 目标事件被误删、运动变得断续：适当增大阈值。
- 建议从 `5000us` 到 `20000us` 之间试起。

## TrailFilterAlgorithm

作用：抑制同一像素上的拖尾事件。它会保留极性变化事件，或者距离上一次同坐标事件足够久的事件。

适合场景：

- 物体边缘后面有明显拖尾。
- 高对比边缘运动后出现残影。
- 想减少同一像素短时间内反复触发的事件。

主要参数：

| 参数 | 当前 UI | 含义 |
| --- | --- | --- |
| `width` | 自动使用相机宽度 | 事件流宽度 |
| `height` | 自动使用相机高度 | 事件流高度 |
| `threshold` | “阈值 (μs)” | 同坐标重复事件的抑制时间窗口，单位微秒 |

调参建议：

- 拖尾仍明显：增大阈值。
- 细节或快速运动被删太多：减小阈值。
- 建议从 `1000us` 到 `10000us` 之间试起。

## SpatioTemporalContrastAlgorithm / STC

作用：利用时空对比过滤错误检测和拖尾。事件需要在给定时间窗口内有前序相关事件才会通过；当前项目启用了 `cut_trail=True`，通过一个事件后，会抑制该像素直到极性变化前的后续事件。

适合场景：

- 背景噪声和拖尾同时存在。
- 需要比 `Activity` 更强的清理效果。
- 事件率过高，希望明显压低输入量。

主要参数：

| 参数 | 当前 UI | 含义 |
| --- | --- | --- |
| `width` | 自动使用相机宽度 | 事件流宽度 |
| `height` | 自动使用相机高度 | 事件流高度 |
| `threshold` | “阈值 (μs)” | STC 时间窗口，单位微秒 |
| `cut_trail` | 固定为 `True` | 是否强力抑制拖尾 |

调参建议：

- 噪声或拖尾仍多：可先减小阈值。
- 目标变稀疏、边缘断裂：增大阈值。
- 建议从 `5000us` 到 `20000us` 之间试起。

## AntiFlickerAlgorithm

作用：去除频闪事件。它针对某个频率区间内反复触发的事件，常用于处理照明频闪。

适合场景：

- 灯光、屏幕、LED、电源频闪造成周期性事件。
- 背景里有大量稳定频率的闪烁噪声。

当前项目默认参数：

| 参数 | 当前设置 | 含义 |
| --- | --- | --- |
| `width` | 自动使用相机宽度 | 事件流宽度 |
| `height` | 自动使用相机高度 | 事件流高度 |
| `filter_length` | `7` | 需要观察到多少次相近周期后才判定频闪 |
| `min_freq` | `50.0` | 要过滤的最低频率，单位 Hz |
| `max_freq` | `70.0` | 要过滤的最高频率，单位 Hz |
| `diff_thresh_us` | “阈值 (μs)” | 连续周期允许的最大时间差，单位微秒 |

调参建议：

- 普通市电照明频闪：先使用默认 `50Hz` 到 `70Hz`。
- 如果是显示器或 LED 控制器频闪，需要把 `min_freq/max_freq` 改到实际频率附近。
- 阈值越大，对周期波动越宽容；太大可能误伤正常周期性运动，太小可能漏掉不稳定频闪。

注意：当前 UI 只暴露阈值，`filter_length/min_freq/max_freq` 固定在
`backend/noise_filter.py` 中。如需调这些参数，需要扩展主界面设置控件和
`NoiseFilterPipeline` 配置。

## 推荐使用顺序

1. 先选 `Activity`，从 `10000us` 开始，看随机噪声是否降低。
2. 如果仍有拖尾，试 `Trail` 或 `STC`。
3. 如果噪声明显来自灯光频闪，试 `AntiFlicker`。
4. 如果滤波后目标变稀疏、推理不稳定，降低滤波强度或回到 `Activity`。

## 常见现象

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 日志显示 `[NoiseFilter] Disabled` | 当前选择是 `None` | 在“去噪算法”下拉框选择具体算法 |
| 日志显示 `Enabled` 但画面变化不明显 | 阈值过宽，或者场景本身噪声少 | 改小或改大阈值，对比事件率和画面 |
| 画面变得断续 | 滤波过强 | 按对应算法的阈值方向减弱过滤，或换成 `Activity` |
| 频闪仍然存在 | 频率不在 `50Hz-70Hz` 内 | 调整 `AntiFlicker` 的 `min_freq/max_freq` |
| 算法初始化失败 | SDK 版本 API 不完全一致 | 优先试 `Activity`、`Trail`、`STC`，查看日志中的具体错误 |

## 参考资料

- Metavision Algorithms Overview: https://docs.prophesee.ai/stable/algorithms.html
- Metavision Noise Filtering Python Sample: https://docs.prophesee.ai/stable/samples/modules/cv/noise_filtering_py.html
- Metavision SDK CV Python API: https://docs.prophesee.ai/stable/api/python/cv/bindings.html
- Metavision Data Rate Viewer: https://docs.prophesee.ai/stable/samples/modules/cv/data_rate_viewer_cpp.html
