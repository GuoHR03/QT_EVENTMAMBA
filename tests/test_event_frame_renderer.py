import numpy as np

from backend.event_frame_renderer import EventFrameRenderer
from backend.event_processing import EVENT_CD_DTYPE


def test_event_frame_renderer_draws_bgr_palette_pixels():
    callbacks = []
    renderer = EventFrameRenderer(
        width=4,
        height=3,
        palette_type="Dark",
        frame_callback=lambda ts, frame: callbacks.append((ts, frame)),
    )
    events = np.array(
        [(1, 1, 1, 100), (2, 1, 0, 200)],
        dtype=EVENT_CD_DTYPE,
    )

    renderer.process_events(events)

    timestamp, frame = callbacks[0]
    assert timestamp == 200
    assert frame.shape == (3, 4, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [52, 37, 30]
    assert frame[1, 1].tolist() == [255, 255, 255]
    assert frame[1, 2].tolist() == [200, 126, 64]


def test_event_frame_renderer_ignores_out_of_bounds_events():
    callbacks = []
    renderer = EventFrameRenderer(
        width=2,
        height=2,
        palette_type="Gray",
        frame_callback=lambda ts, frame: callbacks.append((ts, frame)),
    )
    events = np.array(
        [(10, 10, 1, 100)],
        dtype=EVENT_CD_DTYPE,
    )

    renderer.process_events(events)

    assert callbacks == []


def test_event_frame_renderer_last_event_wins_per_pixel():
    callbacks = []
    renderer = EventFrameRenderer(
        width=3,
        height=3,
        palette_type="Dark",
        frame_callback=lambda ts, frame: callbacks.append((ts, frame)),
    )
    events = np.array(
        [(1, 1, 1, 100), (1, 1, 0, 200)],
        dtype=EVENT_CD_DTYPE,
    )

    renderer.process_events(events)

    assert callbacks[0][1][1, 1].tolist() == [200, 126, 64]


def test_event_frame_renderer_updates_palette_without_rebuild():
    callbacks = []
    renderer = EventFrameRenderer(
        width=3,
        height=3,
        palette_type="Dark",
        frame_callback=lambda ts, frame: callbacks.append((ts, frame)),
    )

    changed = renderer.set_display_settings("Gray", fps=60)
    renderer.process_events(np.array([(1, 1, 1, 100)], dtype=EVENT_CD_DTYPE))

    assert changed is True
    assert callbacks[0][1][0, 0].tolist() == [128, 128, 128]
    assert callbacks[0][1][1, 1].tolist() == [255, 255, 255]


def test_event_frame_renderer_callbacks_receive_stable_frame_copy():
    callbacks = []
    renderer = EventFrameRenderer(
        width=3,
        height=3,
        palette_type="Gray",
        frame_callback=lambda ts, frame: callbacks.append((ts, frame)),
    )

    renderer.process_events(np.array([(1, 1, 1, 100)], dtype=EVENT_CD_DTYPE))
    first_frame = callbacks[0][1]
    renderer.process_events(np.array([(2, 2, 1, 200)], dtype=EVENT_CD_DTYPE))

    assert first_frame[1, 1].tolist() == [255, 255, 255]
    assert first_frame[2, 2].tolist() == [128, 128, 128]
