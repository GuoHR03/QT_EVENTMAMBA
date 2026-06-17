# Version History

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
