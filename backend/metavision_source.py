import logging
import time

from backend.event_processing import filter_events_by_roi, replace_oldest_nowait
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
        anchor_sensor_time = int(self.start_ts or 0)
        anchor_real_time = self.now()
        replay_factor = normalize_replay_factor(self.replay_factor_getter())

        for events in self.iterator:
            target_sensor_time = int(self.iterator.get_current_time())
            anchor_sensor_time, anchor_real_time, replay_factor = self._sleep_until(
                target_sensor_time,
                anchor_sensor_time,
                anchor_real_time,
                replay_factor,
            )
            yield events

    def _sleep_until(self, target_sensor_time, anchor_sensor_time, anchor_real_time, replay_factor):
        while True:
            current_time = self.now()
            current_factor = normalize_replay_factor(self.replay_factor_getter())
            if current_factor != replay_factor:
                return target_sensor_time, current_time, current_factor

            sensor_elapsed_s = (target_sensor_time - anchor_sensor_time) / 1_000_000.0
            real_elapsed_s = current_time - anchor_real_time
            sleep_time = (sensor_elapsed_s / replay_factor) - real_elapsed_s
            if sleep_time <= 0:
                return anchor_sensor_time, anchor_real_time, replay_factor

            self.sleep(min(sleep_time, MAX_DYNAMIC_REPLAY_SLEEP_S))


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
    roi_getter,
    noise_filter,
    frame_generator,
    nn_queue,
    progress_callback=None,
):
    for events in iterator:
        if not is_running():
            break
        if len(events) == 0:
            continue
        if progress_callback is not None:
            progress_callback(int(events["t"][-1]))

        events = filter_events_by_roi(events, roi_getter())
        events = noise_filter.apply(events)
        if len(events) == 0:
            continue

        frame_generator.process_events(events)
        replace_oldest_nowait(nn_queue, events)


def _report(status_callback, message):
    LOGGER.info(message)
    if status_callback is not None:
        status_callback(message)
