from app.settings import AppSettings


def test_app_settings_exposes_single_playback_config_snapshot():
    settings = AppSettings()
    initial = settings.playback_config

    settings.update_capture("Gray", 60, 2.0)
    settings.update_noise_filter("activity", 5000)
    settings.update_roi((10, 20, 30, 40))

    assert settings.playback_config is not initial
    assert settings.palette == "Gray"
    assert settings.fps == 60
    assert settings.replay_factor == 2.0
    assert settings.noise_filter_type == "activity"
    assert settings.noise_filter_threshold_us == 5000
    assert settings.roi == (10, 20, 30, 40)
