# 通过队列去成像和预测

import time
import queue
import numpy as np
import dv_processing as dv
import h5py
import cv2
from libs import metavision_hal
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
try:
    from metavision_sdk_base import EventCD
except Exception:
    EventCD = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', 'i1'), ('t', '<i8')])

try:
    from metavision_sdk_cv import (
        ActivityNoiseFilterAlgorithm,
        AntiFlickerAlgorithm,
        SpatioTemporalContrastAlgorithm,
        TrailFilterAlgorithm,
    )
    _METAVISION_CV_IMPORT_ERROR = None
except Exception as exc:
    ActivityNoiseFilterAlgorithm = None
    AntiFlickerAlgorithm = None
    SpatioTemporalContrastAlgorithm = None
    TrailFilterAlgorithm = None
    _METAVISION_CV_IMPORT_ERROR = exc

try:
    from metavision_sdk_cv import FrequencyEstimationConfig
except Exception:
    FrequencyEstimationConfig = None


try:
    EVENT_CD_DTYPE = np.dtype(EventCD)
except TypeError:
    EVENT_CD_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', 'i1'), ('t', '<i8')])
NOISE_FILTER_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "activity": "activity",
    "activity_noise": "activity",
    "activitynoisefilter": "activity",
    "trail": "trail",
    "trail_filter": "trail",
    "stc": "stc",
    "spatio_temporal_contrast": "stc",
    "spatiotemporalcontrast": "stc",
    "anti_flicker": "anti_flicker",
    "antiflicker": "anti_flicker",
    "flicker": "anti_flicker",
}
NOISE_FILTER_DISPLAY_NAMES = {
    "none": "None",
    "activity": "Activity",
    "trail": "Trail",
    "stc": "STC",
    "anti_flicker": "AntiFlicker",
}


def _normalize_noise_filter_type(filter_type):
    key = str(filter_type or "none").strip().lower().replace("-", "_").replace(" ", "_")
    return NOISE_FILTER_ALIASES.get(key, "none")


def _event_field_name(events, candidates):
    names = events.dtype.names or ()
    for name in candidates:
        if name in names:
            return name
    return None


def _to_event_cd(events):
    if events is None:
        return None
    if len(events) == 0:
        return np.empty(0, dtype=EVENT_CD_DTYPE)

    names = events.dtype.names or ()
    time_field = _event_field_name(events, ("t", "timestamp", "ts"))
    polarity_field = _event_field_name(events, ("p", "pol", "polarity"))
    if "x" not in names or "y" not in names or time_field is None or polarity_field is None:
        raise ValueError("events must have x, y, polarity and timestamp fields")

    if events.dtype == EVENT_CD_DTYPE and time_field == "t" and polarity_field == "p":
        return np.ascontiguousarray(events)

    converted = np.empty(len(events), dtype=EVENT_CD_DTYPE)
    converted["x"] = events["x"]
    converted["y"] = events["y"]
    converted["p"] = events[polarity_field]
    converted["t"] = events[time_field]
    return converted


def _event_time_field(events):
    return _event_field_name(events, ("t", "timestamp", "ts"))


def _normalize_roi(roi, src_width, src_height):
    if not roi:
        return None

    x, y, width, height = [int(v) for v in roi]
    if width <= 0 or height <= 0:
        return None

    x1 = max(0, min(src_width - 1, x))
    y1 = max(0, min(src_height - 1, y))
    x2 = max(x1 + 1, min(src_width, x + width))
    y2 = max(y1 + 1, min(src_height, y + height))
    return x1, y1, x2 - x1, y2 - y1


def filter_events_by_roi(events, roi):
    if events is None or len(events) == 0 or not roi:
        return events

    x, y, width, height = roi
    mask = (
        (events['x'] >= x) & (events['x'] < x + width) &
        (events['y'] >= y) & (events['y'] < y + height)
    )
    return events[mask]


def downsample_roi_normalize_events(data_numpy, roi, src_width=640, src_height=480):
    if data_numpy is None or len(data_numpy) == 0 or not roi:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    roi_x, roi_y, roi_width, roi_height = roi
    mask = (
        (data_numpy[:, 0] >= roi_x) & (data_numpy[:, 0] < roi_x + roi_width) &
        (data_numpy[:, 1] >= roi_y) & (data_numpy[:, 1] < roi_y + roi_height)
    )
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    cropped = data_numpy[mask]
    x_values = (cropped[:, 0] - roi_x) / roi_width
    y_values = (cropped[:, 1] - roi_y) / roi_height
    t_values = cropped[:, 2]
    x_values = np.clip(x_values, 0.0, 1.0)
    y_values = np.clip(y_values, 0.0, 1.0)

    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downsample_crop_normalize_events(data_numpy, src_width=640, src_height=480, dst_width=512, dst_height=512):
    """进行下采样+裁剪+归一化,适用于ini30数据集
    裁剪区域: x: [96, 608], y: [-16, 496] (实际有效)
    输出尺寸: 512x512
    """
    if data_numpy is None or len(data_numpy) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    x_raw = data_numpy[:, 0] * (640.0 / src_width)
    y_raw = data_numpy[:, 1] * (480.0 / src_height)
    x_raw = np.clip(x_raw, 0, 640 - 1)
    y_raw = np.clip(y_raw, 0, 480 - 1)

    # 裁剪 512 * 512
    mask = (x_raw >= 96) & (x_raw <= 608)
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    x_values = x_raw[mask] - 96
    y_values = y_raw[mask] + 16
    t_values = data_numpy[:, 2][mask]

    x_values = np.clip(x_values, 0, dst_width - 1)
    y_values = np.clip(y_values, 0, dst_height - 1)

    # 归一化
    x_values = x_values / dst_width
    y_values = y_values / dst_height
    t_max = t_values.max()
    t_min = t_values.min()
    t_values = (t_values - t_min) / (t_max - t_min + 1e-5)
    t_values = t_values * 0.1
    return x_values, y_values, t_values


def downsample_normalize_events(data_numpy, src_width=640, src_height=480, dst_width=640, dst_height=480):
    """进行下采样+归一化，适用于seet数据集
    输出尺寸: dst_width x dst_height
    """
    if data_numpy is None or len(data_numpy) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

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


def _put_latest(target_queue, payload):
    if target_queue is None:
        return

    try:
        if target_queue.full():
            existing = target_queue.get_nowait()
            if isinstance(existing, dict) and existing.get("msg_type") == "CONFIG":
                target_queue.put_nowait(existing)
                return
    except queue.Empty:
        pass
    except queue.Full:
        return

    try:
        target_queue.put_nowait(payload)
    except queue.Full:
        pass


class NNWorker(QThread):
    """神经网络推理线程 - 独立按 nn_interval 发送推理"""
    finished_signal = pyqtSignal()

    def __init__(self, nn_queue, nn_interval_us, width, height, target_queue, analysis_enabled, roi=None):
        super().__init__()
        self.nn_queue = nn_queue
        self.nn_interval_us = nn_interval_us
        self.width = width
        self.height = height
        self.target_queue = target_queue
        self.analysis_enabled = analysis_enabled
        self.roi = roi
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
                        timestamp = int(nn_events[-1, 2])

                        if self.roi:
                            x_norm, y_norm, t_norm = downsample_roi_normalize_events(
                                nn_events, self.roi, src_width=self.width, src_height=self.height
                            )
                        else:
                            x_norm, y_norm, t_norm = downsample_crop_normalize_events(
                                 nn_events, src_width=self.width, src_height=self.height
                            )

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

                        clean_array = np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)

                        _put_latest(self.target_queue, {
                            "msg_type": "EVENTS",
                            "data": clean_array,
                            "timestamp": timestamp,
                            "cropped": True,
                        })
                    except Exception as exc:
                        print(f"NNWorker error: {exc}")

                buffer = []
                next_nn_time += self.nn_interval_us

        self.finished_signal.emit()


class CameraThread(QThread):
    """事件读取线程 - 读取事件并分发放到两个队列"""
    finished_signal = pyqtSignal()

    def __init__(
        self,
        palette_type="Dark",
        fps=30,
        nn_interval_ms=20,
        target_queue=None,
        file_path="",
        roi=None,
        noise_filter_type="none",
        noise_filter_threshold_us=10000,
    ):
        super().__init__()
        self.is_running = True
        self.is_recording = False
        self.target_queue = target_queue
        self.analysis_enabled = True
        self.input_path = file_path
        self.palette_type = palette_type
        self.fps = fps if fps > 0 else 30
        self.nn_interval_us = int(nn_interval_ms * 1000)
        self.replay_factor = 1.0
        self.width = 640
        self.height = 480
        self.noise_filter_type = _normalize_noise_filter_type(noise_filter_type)
        self.noise_filter_threshold_us = max(1, int(noise_filter_threshold_us or 10000))
        self.noise_filter = None
        self.noise_filter_output = None
        self._noise_filter_warning_printed = False

        self.is_aedat4 = self.input_path and self.input_path.lower().endswith(".aedat4")
        self.is_h5 = self.input_path and self.input_path.lower().endswith(('.h5', '.hdf5'))

        self.nn_queue = queue.Queue(maxsize=10)

        self.nn_worker = None

        self.requested_roi = roi
        self.roi = None
        self.roi_x = None
        self.roi_y = None
        self.roi_width = None
        self.roi_height = None

        self._init_engine(palette_type)

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
            except Exception:
                self.width, self.height = 640, 480
            self._set_roi(self.requested_roi)

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
            self._set_roi(self.requested_roi)

            palette_map = {
                "Dark": ColorPalette.Dark, "Light": ColorPalette.Light,
                "CoolWarm": ColorPalette.CoolWarm, "Gray": ColorPalette.Gray
            }
            palette = palette_map.get(palette_type, ColorPalette.Dark)
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, fps=self.fps, palette=palette)
            self.event_frame_gen.set_output_callback(self._on_cd_frame_cb)

        else:
            self.device = None
            self._set_roi(self.requested_roi)

            if self.input_path:
                self.device = None
                print("[Camera] 使用文件模式回放")
                base_iterator = EventsIterator(input_path=self.input_path, delta_t=self.nn_interval_us)
                self.mv_iterator = LiveReplayEventsIterator(base_iterator, replay_factor=self.replay_factor)
            else:
                try:
                    self.device = initiate_device("")
                except Exception as e:
                    print(f"[Camera] 连接相机失败: {e}")

                if self.device is not None:
                    i_roi = self.device.get_i_roi()
                    if i_roi is not None and self.roi_x is not None:
                        print("[ROI] 设备支持ROI，正在设置...")
                        try:
                            roi_window = metavision_hal.I_ROI.Window(
                                self.roi_x, self.roi_y,
                                self.roi_x + self.roi_width, self.roi_y + self.roi_height
                            )
                            i_roi.set_window(roi_window)
                            i_roi.enable(True)
                            print(f"[ROI] 已设置 ROI: x={self.roi_x}, y={self.roi_y}, width={self.roi_width}, height={self.roi_height}")
                        except Exception as e:
                            print(f"[ROI] 设置ROI失败，异常: {e}")
                    else:
                        if self.roi_x is None:
                            print("[ROI] 未设置ROI参数，跳过硬件ROI设置")
                        else:
                            print("[ROI] 设备不支持ROI，跳过设置")
                    self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=self.nn_interval_us)
                else:
                    print("[Camera] 无法连接相机，也无输入文件")
                    return

            self.height, self.width = self.mv_iterator.get_size()
            self._set_roi(self.requested_roi)

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
            self.target_queue, lambda: self.analysis_enabled, self._roi_tuple()
        )
        self.nn_worker.start()

    def _on_worker_finished(self):
        self.is_running = False

    def _set_roi(self, roi):
        normalized = _normalize_roi(roi, self.width, self.height)
        self.roi = normalized
        if normalized:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = normalized
        else:
            self.roi_x = None
            self.roi_y = None
            self.roi_width = None
            self.roi_height = None

    def _roi_tuple(self):
        if self.roi_x is None:
            return None
        return self.roi_x, self.roi_y, self.roi_width, self.roi_height

    def _init_noise_filter(self):
        if self.noise_filter_type == "none":
            self._report_status("[NoiseFilter] Disabled")
            return
        if _METAVISION_CV_IMPORT_ERROR is not None:
            self._report_status(f"[NoiseFilter] Metavision CV unavailable: {_METAVISION_CV_IMPORT_ERROR}")
            self.noise_filter_type = "none"
            return

        try:
            if self.noise_filter_type == "activity":
                self.noise_filter = ActivityNoiseFilterAlgorithm(
                    self.width, self.height, self.noise_filter_threshold_us
                )
            elif self.noise_filter_type == "trail":
                self.noise_filter = TrailFilterAlgorithm(
                    self.width, self.height, self.noise_filter_threshold_us
                )
            elif self.noise_filter_type == "stc":
                self.noise_filter = SpatioTemporalContrastAlgorithm(
                    self.width, self.height, self.noise_filter_threshold_us, True
                )
            elif self.noise_filter_type == "anti_flicker":
                self.noise_filter = self._create_anti_flicker_filter()
            else:
                self.noise_filter_type = "none"
                return

            self.noise_filter_output = self.noise_filter.get_empty_output_buffer()
            self._report_status(
                "[NoiseFilter] Enabled "
                f"{NOISE_FILTER_DISPLAY_NAMES[self.noise_filter_type]} "
                f"(threshold={self.noise_filter_threshold_us}us)"
            )
        except Exception as exc:
            self._report_status(f"[NoiseFilter] Failed to initialize {self.noise_filter_type}: {exc}")
            self.noise_filter_type = "none"
            self.noise_filter = None
            self.noise_filter_output = None

    def _create_anti_flicker_filter(self):
        if AntiFlickerAlgorithm is None:
            raise RuntimeError("AntiFlickerAlgorithm is unavailable")

        if FrequencyEstimationConfig is not None:
            flicker_config = FrequencyEstimationConfig()
            for attr, value in (
                ("filter_length", 7),
                ("min_freq", 50.0),
                ("max_freq", 70.0),
                ("diff_thresh_us", self.noise_filter_threshold_us),
                ("threshold", self.noise_filter_threshold_us),
            ):
                if hasattr(flicker_config, attr):
                    setattr(flicker_config, attr, value)
            try:
                return AntiFlickerAlgorithm(self.width, self.height, flicker_config)
            except TypeError:
                pass

        constructor_attempts = (
            (self.width, self.height, 7, 50.0, 70.0, self.noise_filter_threshold_us),
            (self.width, self.height, 50.0, 70.0, self.noise_filter_threshold_us),
            (self.width, self.height, self.noise_filter_threshold_us),
        )
        last_error = None
        for args in constructor_attempts:
            try:
                return AntiFlickerAlgorithm(*args)
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"AntiFlicker constructor is not compatible: {last_error}")

    def _report_status(self, message):
        print(message)
        self.status_signal.emit(message)

    def _apply_noise_filter(self, events):
        if self.noise_filter is None or events is None or len(events) == 0:
            return events

        try:
            event_cd = _to_event_cd(events)
            self.noise_filter.process_events(event_cd, self.noise_filter_output)
            try:
                return self.noise_filter_output.numpy(copy=True)
            except TypeError:
                return self.noise_filter_output.numpy().copy()
        except Exception as exc:
            if not self._noise_filter_warning_printed:
                print(f"[NoiseFilter] Filtering failed, passing raw events through: {exc}")
                self._noise_filter_warning_printed = True
            return events

    def run(self):
        self._init_noise_filter()
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
                frame_events = filter_events_by_roi(frame_events, self._roi_tuple())
                frame_events = self._apply_noise_filter(frame_events)
                if len(frame_events) == 0:
                    next_frame_target_time += frame_interval_us
                    continue

                self.event_frame_gen.process_events(frame_events)

                if self.analysis_enabled and self.target_queue is not None:
                    try:
                        nn_events = np.column_stack((
                            frame_events['x'], frame_events['y'], frame_events['t']
                        ))

                        if self._roi_tuple():
                            x_norm, y_norm, t_norm = downsample_roi_normalize_events(
                                nn_events, self._roi_tuple(), src_width=self.width, src_height=self.height
                            )
                            is_cropped = True
                        else:
                            x_norm, y_norm, t_norm = downsample_normalize_events(
                                nn_events, src_width=self.width, src_height=self.height
                            )
                            is_cropped = False

                        target_points = 1024
                        if len(x_norm) >= target_points:
                            indices = np.random.choice(len(x_norm), target_points, replace=False)
                            x_norm = x_norm[indices]
                            y_norm = y_norm[indices]
                            t_norm = t_norm[indices]

                            clean_array = np.column_stack((t_norm, x_norm, y_norm)).astype(np.float32)
                            _put_latest(self.target_queue, {
                                "msg_type": "EVENTS",
                                "data": clean_array,
                                "timestamp": int(frame_events['t'][-1]),
                                "cropped": is_cropped,
                            })
                    except Exception as exc:
                        print(f"H5 inference enqueue error: {exc}")

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
        # ======== 2. 神经网络变量 (精准 20.0ms) ========
        frame_buffer = dv.EventStore()
        next_frame_time = None
        frame_interval_us = int(1_000_000 / self.fps)

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

            frame_buffer.add(events)
            arr = events.numpy()
            
            # 初始化所有秒表
            if start_sensor_time is None:
                start_sensor_time = arr['timestamp'][0]
                next_frame_time = start_sensor_time + frame_interval_us
                next_nn_time = start_sensor_time + self.nn_interval_us
                start_real_time = time.perf_counter()

            arr_for_nn = filter_events_by_roi(arr, self._roi_tuple())
            if len(arr_for_nn) > 0:
                nn_buffer.append(arr_for_nn)

            # ---------------------------------------------------------
            # 浠诲姟 A锛歎I 鐢婚潰娓叉煋 (鏀掑甯х巼瀵瑰簲鐨勬椂闂村嚭鍥?
            # ---------------------------------------------------------
            if arr['timestamp'][-1] >= next_frame_time:
                image_bgr = self.dv_visualizer.generateImage(frame_buffer)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                if self.is_running:
                    self.image_signal.emit(image_rgb.copy(), int(arr['timestamp'][-1]))

                frame_buffer = dv.EventStore()
                next_frame_time += frame_interval_us

            # ---------------------------------------------------------
            # 任务 B：神经网络精准切割 (严格按 20ms 步进)
            # ---------------------------------------------------------
            if arr['timestamp'][-1] >= next_nn_time:
                # 把零碎的数据拼成大段
                if not nn_buffer:
                    next_nn_time += self.nn_interval_us
                    continue

                buffer_events = np.concatenate(nn_buffer)
                time_field = _event_time_field(buffer_events)
                if time_field is None:
                    print("[Camera] aedat4 events do not contain a timestamp field")
                    nn_buffer = []
                    next_nn_time += self.nn_interval_us
                    continue

                # 只要剩余数据的最新时间戳 >= 20ms 的目标时间，就一直切！
                while len(buffer_events) > 0 and buffer_events[time_field][-1] >= next_nn_time:
                    # 使用 searchsorted 找到刚好等于或略微超过 20ms 的那条数据索引
                    split_idx = np.searchsorted(buffer_events[time_field], next_nn_time)
                    
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

                    # 更新网络预测的下一个 20ms 目标
                    next_nn_time += self.nn_interval_us

                # 把剩下的尾巴重新塞回缓冲桶
                nn_buffer = [buffer_events] if len(buffer_events) > 0 else []

        # 读取完毕后，稍微等待 NNWorker 消化完队列里的最后一批数据
        while not self.nn_queue.empty() and self.is_running:
            time.sleep(0.05)

    def _run_metavision_loop(self):
        for evs in self.mv_iterator:
            if not self.is_running:
                break

            if len(evs) == 0:
                continue

            evs = filter_events_by_roi(evs, self._roi_tuple())
            evs = self._apply_noise_filter(evs)
            if len(evs) == 0:
                continue

            self.event_frame_gen.process_events(evs)

            try:
                self.nn_queue.put_nowait(evs)
            except queue.Full:
                try:
                    self.nn_queue.get_nowait()
                    self.nn_queue.put_nowait(evs)
                except (queue.Empty, queue.Full):
                    pass

    def stop(self):
        self.is_running = False

    def update_roi(self, roi):
        """动态更新 ROI 参数（需要重启相机才能生效）"""
        self.requested_roi = roi
        self._set_roi(roi)
        if self._roi_tuple():
            print(f"[Camera] ROI 已更新: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}")
        else:
            print("[Camera] ROI 已清除")

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

    image_signal = pyqtSignal(object, int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
