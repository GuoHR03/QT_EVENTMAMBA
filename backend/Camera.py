# 通过队列去成像和预测


import time
import queue
import cv2
import numpy as np
import dv_processing as dv
import h5py
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from metavision_core.event_io.raw_reader import initiate_device
from libs import metavision_hal
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette


def downSampling_cropping_and_normalization(data_numpy, src_width=640, src_height=480, dst_width=512, dst_height=512):
    """进行下采样+裁剪+归一化,适用于ini30数据集
    裁剪区域: x: [96, 608], y: [-16, 496] (实际有效)
    输出尺寸: 512x512
    """
    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    mask = (x_raw >= 96) & (x_raw <= 608)
    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    # 时间轴缩放
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downSampling_and_normalization(data_numpy, src_width=640, src_height=480, dst_width=640, dst_height=480):
    """进行下采样+归一化，适用于seet数据集
    输出尺寸: dst_width x dst_height
    """
    x_values = data_numpy[:, 0] * (dst_width / src_width)
    y_values = data_numpy[:, 1] * (dst_height / src_height)
    t_values = data_numpy[:, 2]
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)
    x_values = x_values / dst_width
    y_values = y_values / dst_height

    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downSampling_cropping_and_normalization_raw(data_numpy, src_width=1280, src_height=720, dst_width=512, dst_height=512):
    """进行下采样+裁剪+归一化，适用于RAW 1280x720数据集
    1280x720 → 映射到 640x480 → x裁剪 [96, 608], y偏移 +16
    输出尺寸: 512x512
    """
    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    mask = (x_raw >= 96) & (x_raw <= 608)
    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]
    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


class NNWorker(QThread):
    """神经网络推理线程 - 独立按 nn_interval 发送推理"""
    finished_signal = pyqtSignal()

    def __init__(self, nn_queue, nn_interval_us, width, height, target_queue, analysis_enabled):
        super().__init__()
        self.nn_queue = nn_queue
        self.nn_interval_us = nn_interval_us
        self.width = width
        self.height = height
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled
        self.is_running = True

    def run(self):
        buffer = []
        next_nn_time = None

        while self.is_running:
            try:
                events = self.nn_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            if 'timestamp' in events.dtype.names and 't' not in events.dtype.names:
                # 转换为t字段
                old_dtype = events.dtype
                new_fields = []
                for name in old_dtype.names:
                    field_type = old_dtype.fields[name][0]
                    new_name = 't' if name == 'timestamp' else name
                    new_fields.append((new_name, field_type))
                new_dtype = np.dtype(new_fields)
                new_events = np.zeros(len(events), dtype=new_dtype)
                for name in old_dtype.names:
                    new_name = 't' if name == 'timestamp' else name
                    new_events[new_name] = events[name]
                events = new_events

            buffer.append(events)

            if next_nn_time is None:
                next_nn_time = events['t'][-1] + self.nn_interval_us
                continue

            if events['t'][-1] >= next_nn_time:
                if buffer and self.target_queue is not None and self.analysis_enabled():
                    try:
                        nn_events = np.concatenate(buffer)
                        nn_events = np.column_stack((nn_events['x'], nn_events['y'], nn_events['t']))

                        x_norm, y_norm, t_norm = downSampling_cropping_and_normalization_raw(
                             nn_events, src_width=self.width, src_height=self.height
                        )
                        # x_norm, y_norm, t_norm = downSampling_cropping_and_normalization(
                        #      nn_events, src_width=self.width, src_height=self.height
                        # )

                        target_points = 1024
                        if len(x_norm) < target_points:
                            buffer = []
                            next_nn_time += self.nn_interval_us
                            continue
                        if len(x_norm) > target_points:
                            indices = np.random.choice(len(x_norm), target_points, replace=False)
                            x_norm = x_norm[indices]
                            y_norm = y_norm[indices]
                            t_norm = t_norm[indices]
                            # t_norm = t_norm * 0.1
                            
                        clean_array = np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)

                        if self.target_queue.full():
                            self.target_queue.get_nowait()
                        self.target_queue.put_nowait({
                            "msg_type": "EVENTS",
                            "data": clean_array,
                        })
                    except queue.Full:
                        pass

                buffer = []
                next_nn_time += self.nn_interval_us

        self.finished_signal.emit()


class CameraThread(QThread):
    """事件读取线程 - 读取事件并分发放到两个队列"""
    finished_signal = pyqtSignal()

    def __init__(self, palette_type="Dark", fps=30, nn_interval_ms=30, target_queue=None, file_path="", roi_params=None):
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.target_queue = target_queue
        self.analysis_enabled = True
        self.input_path = file_path
        self.palette_type = palette_type
        self.fps = fps if fps > 0 else 30
        self.nn_interval_us = int(nn_interval_ms * 1000)
        self.width = 640
        self.height = 480

        self.is_aedat4 = self.input_path and self.input_path.lower().endswith(".aedat4")
        self.is_h5 = self.input_path and self.input_path.lower().endswith(('.h5', '.hdf5'))

        self.nn_queue = queue.Queue(maxsize=10)

        self.nn_worker = None

        if roi_params:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = roi_params
        else:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = 100, 100, 512, 512

        self._init_engine(palette_type)

    def set_roi(self, roi_x, roi_y, roi_width, roi_height):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        if self.device is not None:
            i_roi = self.device.get_i_roi()
            if i_roi is not None:
                try:
                    roi_window = metavision_hal.I_ROI.Window(roi_x, roi_y, roi_width, roi_height)
                    i_roi.set_window(roi_window)
                    i_roi.enable(True)
                    print(f"[ROI] 动态更新 ROI: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
                except Exception as e:
                    print(f"[ROI] 动态更新失败: {e}")

    def _on_cd_frame_cb(self, ts, frame):
        self.image_signal.emit(frame.copy(), int(ts))

    def _init_engine(self, palette_type):
        if self.is_aedat4:
            # aedat4 调色盘初始化
            self.device = None
            self.dv_reader = dv.io.MonoCameraRecording(self.input_path)

            try:
                res = self.dv_reader.getEventResolution()
                self.width, self.height = res.width, res.height
            except:
                self.width, self.height = 640, 480

            self.dv_visualizer = dv.visualization.EventVisualizer((self.width, self.height))

            palette_rgb_map = {
                "Dark": {
                    "bg": (30, 37, 52),
                    "pos": (255, 255, 255),
                    "neg": (64, 126, 200)
                },
                "Light": {
                    "bg": (255, 255, 255),
                    "pos": (64, 126, 200),
                    "neg": (30, 37, 52)
                },
                "CoolWarm": {
                    "bg": (217, 224, 237),
                    "pos": (255, 113, 117),
                    "neg": (87, 123, 198)
                },
                "Gray": {
                    "bg": (128, 128, 128),
                    "pos": (255, 255, 255),
                    "neg": (0, 0, 0)
                }
            }
            rgb = palette_rgb_map.get(palette_type, palette_rgb_map["Dark"])
            self.dv_visualizer.setBackgroundColor(rgb["bg"])
            self.dv_visualizer.setPositiveColor(rgb["pos"])
            self.dv_visualizer.setNegativeColor(rgb["neg"])

        elif self.is_h5:
            self.device = None
            self.h5_file = h5py.File(self.input_path, 'r')
            self.events_dataset = self.h5_file['events']
            self.h5_dtypes = self.events_dataset.dtype.names

            self.width = self.h5_file.attrs.get('width', 640)
            self.height = self.h5_file.attrs.get('height', 480)

            palette_map = {
                "Dark": ColorPalette.Dark, "Light": ColorPalette.Light,
                "CoolWarm": ColorPalette.CoolWarm, "Gray": ColorPalette.Gray
            }
            palette = palette_map.get(palette_type, ColorPalette.Dark)
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, fps=self.fps, palette=palette)
            self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

        else:
            # metavision 调色盘初始化
            self.device = None
            try:
                self.device = initiate_device("")
            except Exception as e:
                print(f"[Camera] 连接相机失败: {e}")

            if self.device is not None:
                i_roi = self.device.get_i_roi()
                if i_roi is not None:
                    print("[ROI] 设备支持ROI，正在设置...")
                    self.roi_x = 100
                    self.roi_y = 100
                    self.roi_width = 512
                    self.roi_height = 512
                    try:
                        roi_window = metavision_hal.I_ROI.Window(self.roi_x, self.roi_y, self.roi_width, self.roi_height)
                        i_roi.set_window(roi_window)
                        i_roi.enable(True)
                        print(f"[ROI] 已设置 ROI: x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height}")
                    except Exception as e:
                        print(f"[ROI] 设置ROI失败，异常: {e}")
                        self.device = None
                else:
                    print("[ROI] 设备不支持ROI，跳过设置")

            if self.device is not None:
                self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=self.nn_interval_us)
            else:
                print("[Camera] 使用文件模式回放")
                self.mv_iterator = EventsIterator(input_path=self.input_path, delta_t=self.nn_interval_us)

            self.height, self.width = self.mv_iterator.get_size()

            palette_map = {
                "Dark": ColorPalette.Dark, "Light": ColorPalette.Light,
                "CoolWarm": ColorPalette.CoolWarm, "Gray": ColorPalette.Gray
            }
            palette = palette_map.get(palette_type, ColorPalette.Dark)
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, fps=self.fps, palette=palette)
            self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

    def _start_workers(self, palette_type):
        self.nn_worker = NNWorker(
            self.nn_queue, self.nn_interval_us, self.width, self.height,
            self.target_queue, lambda: self.analysis_enabled
        )
        self.nn_worker.start()

    def _on_worker_finished(self):
        self.is_running = False

    def run(self):
        self._start_workers(self.palette_type)

        if self.is_aedat4:
            self._run_aedat4_loop()
        elif self.is_h5:
            self._run_h5_loop()
        else:
            self._run_metavision_loop()

        self.is_running = False
        if self.nn_worker:
            self.nn_worker.is_running = False
            self.nn_worker.wait()

        self.finished_signal.emit()

    def _run_h5_loop(self):
        """ 需要大量修改"""
        total_events = len(self.events_dataset)
        current_idx = 0

        time_key = 't' if 't' in self.h5_dtypes else ('ts' if 'ts' in self.h5_dtypes else 'timestamp')
        pol_key = 'p' if 'p' in self.h5_dtypes else ('pol' if 'pol' in self.h5_dtypes else 'polarity')

        frame_interval_us = int(1_000_000 / self.fps)
        start_real_time = time.perf_counter()
        start_sensor_time = None
        next_frame_target_time = None

        while self.is_running and current_idx < total_events:
            events_for_this_frame = []

            while current_idx < total_events:
                step = 5000
                end_idx = min(current_idx + step, total_events)
                raw_events = self.events_dataset[current_idx:end_idx]

                evs = np.zeros(len(raw_events), dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i1'), ('t', '<i8')])
                evs['x'] = raw_events['x']
                evs['y'] = raw_events['y']
                evs['p'] = raw_events[pol_key]
                evs['t'] = raw_events[time_key]

                if start_sensor_time is None:
                    start_sensor_time = evs['t'][0]
                    next_frame_target_time = start_sensor_time + frame_interval_us
                    start_real_time = time.perf_counter()

                over_time_indices = np.where(evs['t'] >= next_frame_target_time)[0]

                if len(over_time_indices) > 0:
                    split_idx = over_time_indices[0]
                    events_for_this_frame.append(evs[:split_idx])
                    current_idx = current_idx + split_idx
                    break
                else:
                    events_for_this_frame.append(evs)
                    current_idx = end_idx

            if not events_for_this_frame:
                break

            frame_events = np.concatenate(events_for_this_frame)

            if len(frame_events) > 0:
                self.event_frame_gen.process_events(frame_events)

                if self.analysis_enabled and self.target_queue is not None:
                    try:
                        nn_events = np.column_stack((
                            frame_events['x'], frame_events['y'], frame_events['t']
                        ))

                        x_norm, y_norm, t_norm = downSampling_and_normalization(
                            nn_events, src_width=self.width, src_height=self.height
                        )

                        target_points = 1024
                        if len(x_norm) >= target_points:
                            indices = np.random.choice(len(x_norm), target_points, replace=False)
                            x_norm = x_norm[indices]
                            y_norm = y_norm[indices]
                            t_norm = t_norm[indices]

                            clean_array = np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)
                            if self.target_queue.full():
                                self.target_queue.get_nowait()
                            self.target_queue.put_nowait({
                                "msg_type": "EVENTS",
                                "data": clean_array,
                            })
                    except queue.Full:
                        pass

            next_frame_target_time += frame_interval_us
            sensor_elapsed_s = (next_frame_target_time - start_sensor_time) / 1_000_000.0
            real_elapsed_s = time.perf_counter() - start_real_time
            sleep_time = sensor_elapsed_s - real_elapsed_s

            if sleep_time > 0.005:
                time.sleep(sleep_time)
            elif sleep_time < -0.2:
                start_real_time = time.perf_counter()
                start_sensor_time = next_frame_target_time - frame_interval_us

        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def _run_aedat4_loop(self):
    # 第一帧不受fps间隔限制，而是第一个事件包里面所有事件进行成像处理
        # ======== 1. UI 渲染变量 (约 33.3ms) ========
        frame_buffer = dv.EventStore()
        next_frame_time = None
        frame_interval_us = int(1_000_000 / self.fps)
        
        # ======== 2. 神经网络变量 (精准 20.0ms) ========
        nn_buffer = []
        next_nn_time = None

        # ======== 3. 真实播放速度控制变量 ========
        start_real_time = time.perf_counter()
        start_sensor_time = None

        while self.is_running and self.dv_reader.isRunning():
            events = self.dv_reader.getNextEventBatch()

            if events is None:
                break
            if events.isEmpty():
                continue

            # aedat4 的专有格式：EventStore 喂给画面，Numpy 喂给网络
            frame_buffer.add(events)
            arr = events.numpy()
            
            # 初始化所有秒表
            if start_sensor_time is None:
                start_sensor_time = arr['timestamp'][0]
                next_frame_time = start_sensor_time + frame_interval_us
                next_nn_time = start_sensor_time + self.nn_interval_us
                start_real_time = time.perf_counter()

            nn_buffer.append(arr)

            # ---------------------------------------------------------
            # 任务 A：UI 画面渲染 (攒够帧率对应的时间出图)
            # ---------------------------------------------------------
            if arr['timestamp'][-1] >= next_frame_time:
                image_bgr = self.dv_visualizer.generateImage(frame_buffer)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                if self.is_running:
                    self.image_signal.emit(image_rgb.copy(), int(arr['timestamp'][-1]))

                frame_buffer = dv.EventStore()
                # 更新下一帧的目标时间
                next_frame_time = arr['timestamp'][-1] + frame_interval_us
                # next_frame_time += frame_interval_us

            # ---------------------------------------------------------
            # 任务 B：神经网络精准切割 (严格按 20ms 步进)
            # ---------------------------------------------------------
            if arr['timestamp'][-1] >= next_nn_time:
                # 把零碎的数据拼成大段
                buffer_events = np.concatenate(nn_buffer)

                # 只要剩余数据的最新时间戳 >= 20ms 的目标时间，就一直切！
                while len(buffer_events) > 0 and buffer_events['timestamp'][-1] >= next_nn_time:
                    # 使用 searchsorted 找到刚好等于或略微超过 20ms 的那条数据索引
                    split_idx = np.searchsorted(buffer_events['timestamp'], next_nn_time)
                    
                    # 🔪 手起刀落，切出极其精准的 20ms 事件段！
                    nn_chunk = buffer_events[:split_idx]

                    if len(nn_chunk) > 0 and self.analysis_enabled and self.nn_queue is not None:
                        try:
                            # 阻塞模式，死等 NNWorker，坚决不丢弃一丝眼动数据！
                            self.nn_queue.put(nn_chunk, timeout=1.0)
                        except queue.Full:
                            print("警告：aedat4 回放中，NNWorker 矩阵运算速度滞后！")

                    # 切割剩下的数据，留给下一次循环
                    buffer_events = buffer_events[split_idx:]
                    
                    # ======== 任务 C：真实时间同步 (在切割点控制速度) ========
                    sensor_elapsed_s = (next_nn_time - start_sensor_time) / 1_000_000.0
                    real_elapsed_s = time.perf_counter() - start_real_time
                    sleep_time = sensor_elapsed_s - real_elapsed_s

                    if sleep_time > 0.005:
                        time.sleep(sleep_time) # 播太快了，等一下真实世界
                    elif sleep_time < -0.2:
                        # 电脑太卡严重滞后，重新对齐时间轴
                        start_real_time = time.perf_counter()
                        start_sensor_time = next_nn_time
                        next_frame_time = start_sensor_time + frame_interval_us

                    # 更新网络预测的下一个 20ms 目标
                    next_nn_time += self.nn_interval_us

                # 把剩下的尾巴重新塞回缓冲桶
                nn_buffer = [buffer_events] if len(buffer_events) > 0 else []

        # 读取完毕后，稍微等待 NNWorker 消化完队列里的最后一批数据
        while not self.nn_queue.empty() and self.is_running:
            time.sleep(0.05)

    def _run_metavision_loop(self):
        start_sensor_time = None
        start_real_time = None

        for evs in self.mv_iterator:
            if not self.is_running:
                break

            if len(evs) == 0:
                continue

            if start_sensor_time is None:
                start_sensor_time = evs['t'][0]
                start_real_time = time.perf_counter()
            else:
                sensor_elapsed = (evs['t'][-1] - start_sensor_time) / 1_000_000.0
                real_elapsed = time.perf_counter() - start_real_time
                min_delay = sensor_elapsed - real_elapsed
                if min_delay > 0:
                    time.sleep(min_delay)

            self.event_frame_gen.process_events(evs)

            try:
                self.nn_queue.put_nowait(evs)
            except queue.Full:
                try:
                    self.nn_queue.get_nowait()
                    self.nn_queue.put_nowait(evs)
                except:
                    pass

    def stop(self):
        self.is_running = False

    def start_recording(self):
        if self.device is not None:
            i_events_stream = self.device.get_i_events_stream()
            if i_events_stream:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                i_events_stream.start_log_raw_data(f"recording_{timestamp}.raw")
                self.is_recording = True

    def stop_recording(self):
        if self.device is not None:
            i_events_stream = self.device.get_i_events_stream()
            if i_events_stream:
                i_events_stream.stop_log_raw_data()
                self.is_recording = False

    image_signal = pyqtSignal(object, object)
    finished_signal = pyqtSignal()  