from PyQt6.QtCore import QSettings

from app.preferences import PreferenceStore, UiPreferences


def _store(tmp_path):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    settings.clear()
    return PreferenceStore(settings)


def test_preferences_round_trip_safe_ui_values(tmp_path):
    store = _store(tmp_path)
    expected = UiPreferences(
        palette="Gray",
        fps=60.0,
        replay_speed="2x",
        prediction_mode="ellipse",
        noise_filter_type="activity",
        noise_filter_threshold_us=5000,
        log_collapsed=True,
        settings_panel_visible=True,
    )

    store.save(expected)

    assert store.load() == expected


def test_preferences_reject_unknown_choices_and_bound_numbers(tmp_path):
    store = _store(tmp_path)
    store.settings.setValue("capture/palette", "unknown")
    store.settings.setValue("capture/fps", 99999)
    store.settings.setValue("capture/replay_speed", "100x")
    store.settings.setValue("inference/prediction_mode", "invalid")
    store.settings.setValue("processing/noise_threshold_us", -1)

    preferences = store.load()

    assert preferences.palette == "Dark"
    assert preferences.fps == 999.99
    assert preferences.replay_speed == "1x"
    assert preferences.prediction_mode == "center"
    assert preferences.noise_filter_threshold_us == 1


def test_preferences_remember_existing_file_directories(tmp_path):
    store = _store(tmp_path)
    input_file = tmp_path / "events" / "sample.raw"
    model_file = tmp_path / "models" / "model.onnx"
    input_file.parent.mkdir()
    model_file.parent.mkdir()

    store.remember_input_file(input_file)
    store.remember_model_file(model_file)

    assert store.last_input_directory(tmp_path) == str(input_file.parent)
    assert store.last_model_directory(tmp_path) == str(model_file.parent)
