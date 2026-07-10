import logging
import time

import numpy as np

from backend.aedat4_replay import (
    split_next_aedat4_nn_chunk,
)
from backend.event_processing import filter_events_by_roi, replace_oldest_nowait, to_event_cd
from backend.replay_clock import ReplayClock, frame_interval_us

LOGGER = logging.getLogger(__name__)


def run_aedat4_replay_loop(
    reader,
    frame_generator,
    fps,
    nn_interval_us,
    is_running,
    roi_getter,
    nn_queue,
    noise_filter,
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
    first_frame_emitted = False

    nn_buffer = []
    next_nn_time = None

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
            next_nn_time = first_timestamp + nn_interval_us
        else:
            current_frame_start = clock.next_frame_time - clock.frame_interval_us
            clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), current_frame_start)

        display_events = filter_events_by_roi(frame_events, roi_getter())
        display_events = noise_filter.apply(display_events)
        if len(display_events) > 0:
            if not first_frame_emitted:
                frame_generator.process_events(np.ascontiguousarray(display_events))
                first_frame_emitted = True
                clock.reset_origin(display_events["t"][-1], now())
            else:
                frame_buffer.append(display_events)
            nn_buffer.append(display_events)

        if frame_events["t"][-1] >= clock.next_frame_time:
            frame_buffer = _drain_frame_chunks(
                frame_buffer,
                clock,
                is_running,
                frame_generator,
                sleep,
                now,
                replay_factor_getter,
                fps,
                fps_getter,
            )

        if frame_events["t"][-1] >= next_nn_time:
            nn_buffer, next_nn_time = _drain_nn_chunks(
                nn_buffer,
                next_nn_time,
                nn_interval_us,
                is_running,
                nn_queue,
            )

    if frame_buffer and is_running():
        frame_generator.process_events(np.ascontiguousarray(np.concatenate(frame_buffer)))


def _drain_frame_chunks(
    frame_buffer,
    clock,
    is_running,
    frame_generator,
    sleep,
    now,
    replay_factor_getter=None,
    fps=None,
    fps_getter=None,
):
    if not frame_buffer:
        clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), clock.next_frame_time)
        return frame_buffer

    buffer_events = np.concatenate(frame_buffer)
    frame_chunk, buffer_events, time_field = split_next_aedat4_nn_chunk(buffer_events, clock.next_frame_time)
    if time_field is None:
        LOGGER.warning("AEDAT4 events do not contain a timestamp field")
        clock.advance_frame()
        return []

    while frame_chunk is not None:
        clock.sleep_until(
            clock.next_frame_time,
            sleep,
            now,
            reset_sensor_time=clock.next_frame_time,
            replay_factor_getter=replay_factor_getter,
            factor_reset_sensor_time=clock.next_frame_time - clock.frame_interval_us,
        )

        if len(frame_chunk) > 0 and is_running():
            frame_generator.process_events(np.ascontiguousarray(frame_chunk))

        clock.reschedule_next_frame(_active_frame_interval_us(fps, fps_getter), clock.next_frame_time)
        frame_chunk, buffer_events, _ = split_next_aedat4_nn_chunk(buffer_events, clock.next_frame_time)

    return [buffer_events] if len(buffer_events) > 0 else []


def _drain_nn_chunks(
    nn_buffer,
    next_nn_time,
    nn_interval_us,
    is_running,
    nn_queue,
):
    if not nn_buffer:
        return nn_buffer, next_nn_time + nn_interval_us

    buffer_events = np.concatenate(nn_buffer)
    nn_chunk, buffer_events, time_field = split_next_aedat4_nn_chunk(buffer_events, next_nn_time)
    if time_field is None:
        LOGGER.warning("AEDAT4 events do not contain a timestamp field")
        return [], next_nn_time + nn_interval_us

    while nn_chunk is not None:
        if len(nn_chunk) > 0 and is_running() and nn_queue is not None:
            if not replace_oldest_nowait(nn_queue, nn_chunk):
                LOGGER.warning("AEDAT4 playback dropped an NN chunk because the queue is unavailable")

        next_nn_time += nn_interval_us
        nn_chunk, buffer_events, _ = split_next_aedat4_nn_chunk(buffer_events, next_nn_time)

    return [buffer_events] if len(buffer_events) > 0 else [], next_nn_time


def _active_frame_interval_us(fps, fps_getter=None):
    return frame_interval_us(fps_getter() if fps_getter is not None else fps)
