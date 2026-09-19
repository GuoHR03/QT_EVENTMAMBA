# 模型资产目录

- 根目录：程序正式运行和安装包使用的 ONNX 模型及配套矩阵。
- `sources/`：生成正式 native-FPS 模型所需、由版本控制保留的 ONNX 源资产。
- `experimental/`：探针、基准测试和转换过程产生的 ONNX、NPZ 与日志；不进入版本控制或安装包，可随时重新生成。

`manifest.json` 记录正式运行资产、生成源资产、文件大小和 SHA-256。修改或重新
生成任何配套资产时必须同步更新清单，并运行：

```powershell
.\.venv-dev\Scripts\python.exe tools\validate_asset_manifest.py
```

安装包只携带 `runtime_assets` 和清单；`source_assets` 仅用于版本追溯与重新生成，
不会复制到最终安装目录。

界面的 Windows 模型选择器优先显示根目录下的 `*_native_fps.onnx`。导入其他正式模型时，建议使用同样明确的发布命名，并将转换中间产物留在 `sources/` 或 `experimental/`。
