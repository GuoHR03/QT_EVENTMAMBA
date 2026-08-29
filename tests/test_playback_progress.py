from app.playback_progress import (
    PLAYBACK_SLIDER_MAX,
    PlaybackProgressState,
    format_playback_time_us,
)


def test_progress_update_clamps_time_and_maps_slider_value():
    state = PlaybackProgressState()

    view = state.update(current_us=15_000_000, total_us=10_000_000)

    assert view.enabled
    assert view.slider_value == PLAYBACK_SLIDER_MAX
    assert view.label == "00:10.000 / 00:10.000"


def test_drag_preview_does_not_overwrite_slider_and_returns_seek_fraction():
    state = PlaybackProgressState()
    state.update(current_us=1_000_000, total_us=10_000_000)

    assert state.begin_drag()
    background_update = state.update(2_000_000, 10_000_000)
    preview = state.preview(7500)
    seek = state.finish_drag(7500)

    assert background_update.update_slider is False
    assert preview.update_slider is False
    assert preview.label == "00:07.500 / 00:10.000"
    assert seek.fraction == 0.75
    assert seek.view.slider_value == 7500


def test_progress_reset_and_inactive_drag_are_safe():
    state = PlaybackProgressState()

    assert not state.begin_drag()
    assert state.preview(100) is None
    assert state.finish_drag(100) is None
    assert state.reset().label == "--:-- / --:--"


def test_time_formatter_supports_hour_long_recordings():
    assert format_playback_time_us(3_661_250_000) == "1:01:01.250"
