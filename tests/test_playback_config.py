from backend.playback_config import PlaybackConfig, PlaybackConfigController, playback_restart_required


def test_playback_config_normalizes_all_runtime_settings():
    config = PlaybackConfig(
        palette="",
        fps=0,
        replay_factor=-1,
        roi=(1, 2, 30, 40),
        noise_filter_type="STC",
        noise_filter_threshold_us=0,
        nn_interval_ms=20,
    )

    assert config.palette == "Dark"
    assert config.fps == 30
    assert config.replay_factor == 0.001
    assert config.roi == (1, 2, 30, 40)
    assert config.noise_filter_type == "stc"
    assert config.noise_filter_threshold_us == 10000
    assert config.nn_interval_us == 20000


def test_playback_config_update_returns_new_immutable_snapshot():
    initial = PlaybackConfig()
    updated = initial.with_updates(fps=60, roi=(10, 20, 30, 40))

    assert initial.fps == 30
    assert initial.roi is None
    assert updated.fps == 60
    assert updated.roi == (10, 20, 30, 40)


def test_playback_config_controller_swaps_complete_snapshot():
    initial = PlaybackConfig()
    controller = PlaybackConfigController(initial)
    updated = initial.with_updates(noise_filter_type="activity", noise_filter_threshold_us=5000)

    previous = controller.set(updated)

    assert previous is initial
    assert controller.get() is updated


def test_only_live_hardware_roi_or_inference_interval_requires_restart():
    initial = PlaybackConfig()
    roi_update = initial.with_updates(roi=(10, 20, 30, 40))
    noise_update = initial.with_updates(noise_filter_type="activity")
    interval_update = initial.with_updates(nn_interval_ms=25)

    assert playback_restart_required(initial, roi_update, input_path="")
    assert not playback_restart_required(initial, roi_update, input_path="events.raw")
    assert not playback_restart_required(initial, noise_update, input_path="")
    assert playback_restart_required(initial, interval_update, input_path="events.h5")
