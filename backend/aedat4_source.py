import logging
import queue
import time

import cv2
import numpy as np

from backend.aedat4_replay import (
    init_aedat4_timing,
    replay_sleep_s,
    should_emit_frame,
    should_reset_replay_clock,
    split_next_aedat4_nn_chunk,
)
from backend.event_processing import filter_events_by_roi

LOGGER = logging.getLogger(__name__)


def run_aedat4_replay_loop(
    reader,
    visualizer,
    event_store_factory,
    fps,
    nn_interval_us,
    is_running,
    roi_getter,
    image_callback,
    nn_queue,
    sleep=time.sleep,
    now=time.perf_counter,
):
    frame_buffer = event_store_factory()
    frame_interval_us = int(1_000_000 / (fps if fps > 0 else 30))
    next_frame_time = None

    nn_buffer = []
    next_nn_time = None

    start_real_time = now()
    start_sensor_time = None

    while is_running() and reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            break
        if events.isEmpty():
            continue

        frame_buffer.add(events)
        event_array = events.numpy()

        if start_sensor_time is None:
            timing = init_aedat4_timing(
                event_array["timestamp"][0],
                frame_interval_us,
                nn_interval_us,
                now(),
            )
            start_sensor_time = timing["start_sensor_time"]
            next_frame_time = timing["next_frame_time"]
            next_nn_time = timing["next_nn_time"]
            start_real_time = timing["start_real_time"]

        roi_events = filter_events_by_roi(event_array, roi_getter())
        if len(roi_events) > 0:
            nn_buffer.append(roi_events)

        if should_emit_frame(event_array, next_frame_time):
            image_bgr = visualizer.generateImage(frame_buffer)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if is_running():
                image_callback(image_rgb.copy(), int(event_array["timestamp"][-1]))

            frame_buffer = event_store_factory()
            next_frame_time += frame_interval_us

        if event_array["timestamp"][-1] >= next_nn_time:
            nn_buffer, start_sensor_time, start_real_time, next_nn_time = _drain_nn_chunks(
                nn_buffer,
                next_nn_time,
                nn_interval_us,
                start_sensor_time,
                start_real_time,
                is_running,
                nn_queue,
                sleep,
                now,
            )

    while not nn_queue.empty() and is_running():
        sleep(0.05)


def _drain_nn_chunks(
    nn_buffer,
    next_nn_time,
    nn_interval_us,
    start_sensor_time,
    start_real_time,
    is_running,
    nn_queue,
    sleep,
    now,
):
    if not nn_buffer:
        return nn_buffer, start_sensor_time, start_real_time, next_nn_time + nn_interval_us

    buffer_events = np.concatenate(nn_buffer)
    nn_chunk, buffer_events, time_field = split_next_aedat4_nn_chunk(buffer_events, next_nn_time)
    if time_field is None:
        LOGGER.warning("AEDAT4 events do not contain a timestamp field")
        return [], start_sensor_time, start_real_time, next_nn_time + nn_interval_us

    while nn_chunk is not None:
        if len(nn_chunk) > 0 and is_running() and nn_queue is not None:
            try:
                nn_queue.put(nn_chunk, timeout=1.0)
            except queue.Full:
                LOGGER.warning("AEDAT4 playback is waiting for NNWorker")

        sleep_time = replay_sleep_s(next_nn_time, start_sensor_time, start_real_time, now())
        if sleep_time > 0.005:
            sleep(sleep_time)
        elif should_reset_replay_clock(sleep_time):
            start_real_time = now()
            start_sensor_time = next_nn_time

        next_nn_time += nn_interval_us
        nn_chunk, buffer_events, _ = split_next_aedat4_nn_chunk(buffer_events, next_nn_time)

    return [buffer_events] if len(buffer_events) > 0 else [], start_sensor_time, start_real_time, next_nn_time
