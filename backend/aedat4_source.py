import logging
import time

import numpy as np

from backend.aedat4_replay import (
    split_events_at_time,
)
from backend.event_processing import to_event_cd
from backend.replay_clock import ReplayClock, frame_interval_us

LOGGER = logging.getLogger(__name__)

_NO_ROI = object()


def run_aedat4_replay_loop(
    reader,
    event_pipeline,
    fps,
    is_running,
    replay_factor=1.0,
    replay_factor_getter=None,
    fps_getter=None,
    start_time_us=0,
    progress_callback=None,
    sleep=time.sleep,
    now=time.perf_counter,
):
    frame_interval = _active_frame_interval_us(fps, fps_getter)
    frame_buffer = []
    frame_buffer_roi = _NO_ROI

    clock = None

    while is_running() and reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            break
        if events.isEmpty():
            continue

        event_array = events.numpy()
        frame_events = to_event_cd(event_array)
        if len(frame_events) == 0:
            continue
        if start_time_us and int(frame_events["t"][-1]) < int(start_time_us):
            continue
        if start_time_us and int(frame_events["t"][0]) < int(start_time_us):
            frame_events = frame_events[frame_events["t"] >= int(start_time_us)]
            if len(frame_events) == 0:
                continue
        if progress_callback is not None:
            progress_callback(int(frame_events["t"][-1]))

        if clock is None:
            first_timestamp = int(frame_events["t"][0])
            current_time = now()
            active_replay_factor = replay_factor_getter() if replay_factor_getter is not None else replay_factor
            clock = ReplayClock.start(first_timestamp, frame_interval, current_time, active_replay_factor)
        else:
            current_frame_start = clock.next_frame_time - clock.frame_interval_us
            clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), current_frame_start)

        roi_snapshot = event_pipeline.current_roi_snapshot()
        display_events = event_pipeline.process_events(
            frame_events,
            render=False,
            roi_snapshot=roi_snapshot,
        )
        effective_roi = roi_snapshot[1]
        if effective_roi != frame_buffer_roi:
            frame_buffer = []
            frame_buffer_roi = effective_roi
        if len(display_events) > 0:
            frame_buffer.append(display_events)

        if frame_events["t"][-1] >= clock.next_frame_time:
            if not event_pipeline.is_roi_current(frame_buffer_roi):
                frame_buffer = []
                frame_buffer_roi = _NO_ROI
                continue
            frame_buffer = _drain_frame_chunks(
                frame_buffer,
                clock,
                is_running,
                event_pipeline,
                sleep,
                now,
                replay_factor_getter,
                fps,
                fps_getter,
                frame_buffer_roi,
            )

    if (
        frame_buffer
        and is_running()
        and event_pipeline.is_roi_current(frame_buffer_roi)
    ):
        event_pipeline.render_events(np.ascontiguousarray(np.concatenate(frame_buffer)))


def _drain_frame_chunks(
    frame_buffer,
    clock,
    is_running,
    event_pipeline,
    sleep,
    now,
    replay_factor_getter=None,
    fps=None,
    fps_getter=None,
    roi=None,
):
    if not frame_buffer:
        clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), clock.next_frame_time)
        return frame_buffer

    buffer_events = np.concatenate(frame_buffer)
    frame_chunk, buffer_events, time_field = split_events_at_time(buffer_events, clock.next_frame_time)
    if time_field is None:
        LOGGER.warning("AEDAT4 events do not contain a timestamp field")
        clock.advance_frame()
        return []

    while frame_chunk is not None:
        if not event_pipeline.is_roi_current(roi):
            return []
        clock.sleep_until(
            clock.next_frame_time,
            sleep,
            now,
            reset_sensor_time=clock.next_frame_time,
            replay_factor_getter=replay_factor_getter,
            factor_reset_sensor_time=clock.next_frame_time - clock.frame_interval_us,
        )

        if not event_pipeline.is_roi_current(roi):
            return []
        if len(frame_chunk) > 0 and is_running():
            event_pipeline.render_events(frame_chunk)

        clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), clock.next_frame_time)
        frame_chunk, buffer_events, _ = split_events_at_time(buffer_events, clock.next_frame_time)

    return [buffer_events] if len(buffer_events) > 0 else []


def _active_frame_interval_us(fps, fps_getter=None):
    return frame_interval_us(fps_getter() if fps_getter is not None else fps)
