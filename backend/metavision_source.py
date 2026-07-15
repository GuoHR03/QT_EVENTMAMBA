import logging
import time
from threading import Event

from backend.replay_speed import normalize_replay_factor

LOGGER = logging.getLogger(__name__)


MAX_DYNAMIC_REPLAY_SLEEP_S = 0.05


def metavision_replay_factor(speed_factor):
    speed_factor = max(float(speed_factor or 1.0), 0.001)
    return 1.0 / speed_factor


class DynamicReplayEventsIterator:
    def __init__(
        self,
        events_iterator,
        replay_factor=1.0,
        replay_factor_getter=None,
        sleep=time.sleep,
        now=time.perf_counter,
    ):
        self.iterator = events_iterator
        self.replay_factor_getter = replay_factor_getter or (lambda: replay_factor)
        self.sleep = sleep
        self.now = now
        self._stop_requested = Event()

    @property
    def start_ts(self):
        return self.iterator.start_ts

    @property
    def delta_t(self):
        return self.iterator.delta_t

    def get_size(self):
        return self.iterator.get_size()

    def get_current_time(self):
        return self.iterator.get_current_time()

    def __iter__(self):
        if self._stop_requested.is_set():
            return

        anchor_sensor_time = int(self.start_ts or 0)
        anchor_real_time = self.now()
        replay_factor = normalize_replay_factor(self.replay_factor_getter())

        for events in self.iterator:
            if self._stop_requested.is_set():
                break
            target_sensor_time = int(self.iterator.get_current_time())
            anchor_sensor_time, anchor_real_time, replay_factor = self._sleep_until(
                target_sensor_time,
                anchor_sensor_time,
                anchor_real_time,
                replay_factor,
            )
            if self._stop_requested.is_set():
                break
            yield events

    def _sleep_until(self, target_sensor_time, anchor_sensor_time, anchor_real_time, replay_factor):
        while True:
            if self._stop_requested.is_set():
                return anchor_sensor_time, anchor_real_time, replay_factor
            current_time = self.now()
            current_factor = normalize_replay_factor(self.replay_factor_getter())
            if current_factor != replay_factor:
                return target_sensor_time, current_time, current_factor

            sensor_elapsed_s = (target_sensor_time - anchor_sensor_time) / 1_000_000.0
            real_elapsed_s = current_time - anchor_real_time
            sleep_time = (sensor_elapsed_s / replay_factor) - real_elapsed_s
            if sleep_time <= 0:
                return anchor_sensor_time, anchor_real_time, replay_factor

            sleep_duration = min(sleep_time, MAX_DYNAMIC_REPLAY_SLEEP_S)
            if self.sleep is time.sleep:
                self._stop_requested.wait(sleep_duration)
            else:
                self.sleep(sleep_duration)

    def request_stop(self):
        self._stop_requested.set()


def create_metavision_iterator(
    input_path,
    device,
    delta_t_us,
    replay_factor,
    replay_factor_getter=None,
    start_ts=0,
):
    from metavision_core.event_io import EventsIterator

    if input_path:
        LOGGER.info("Using Metavision file replay mode")
        base_iterator = EventsIterator(input_path=input_path, start_ts=int(start_ts or 0), delta_t=delta_t_us)
        return DynamicReplayEventsIterator(
            base_iterator,
            replay_factor=replay_factor,
            replay_factor_getter=replay_factor_getter,
        )
    return EventsIterator.from_device(device=device, delta_t=delta_t_us)


def apply_hardware_roi(device, roi, status_callback=None):
    if device is None:
        return

    x, y, width, height = roi or (None, None, None, None)
    if x is None:
        _report(status_callback, "[ROI] No ROI configured; skipping hardware ROI")
        return

    i_roi = device.get_i_roi()
    if i_roi is None:
        _report(status_callback, "[ROI] Device does not support hardware ROI; skipping")
        return

    from libs import metavision_hal

    _report(status_callback, "[ROI] Hardware ROI is supported; applying ROI")
    roi_window = metavision_hal.I_ROI.Window(x, y, x + width, y + height)
    i_roi.set_window(roi_window)
    i_roi.enable(True)
    _report(status_callback, f"[ROI] Applied ROI: x={x}, y={y}, width={width}, height={height}")


def run_metavision_event_loop(
    iterator,
    is_running,
    event_pipeline,
    progress_callback=None,
):
    for events in iterator:
        if not is_running():
            break
        if len(events) == 0:
            continue
        if progress_callback is not None:
            progress_callback(int(events["t"][-1]))

        event_pipeline.process_events(events)


def _report(status_callback, message):
    LOGGER.info(message)
    if status_callback is not None:
        status_callback(message)
