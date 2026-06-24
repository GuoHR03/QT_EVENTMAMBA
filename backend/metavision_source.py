import logging

from backend.event_processing import filter_events_by_roi, replace_oldest_nowait

LOGGER = logging.getLogger(__name__)


def create_metavision_iterator(input_path, device, delta_t_us, replay_factor):
    from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator

    if input_path:
        LOGGER.info("Using Metavision file replay mode")
        base_iterator = EventsIterator(input_path=input_path, delta_t=delta_t_us)
        return LiveReplayEventsIterator(base_iterator, replay_factor=replay_factor)
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
):
    for events in iterator:
        if not is_running():
            break
        if len(events) == 0:
            continue

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
