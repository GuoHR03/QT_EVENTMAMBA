from types import SimpleNamespace

from app.camera_ui_coordinator import CameraUiCoordinator, CameraUiPorts


class FakeValueWidget:
    def __init__(self, value=None):
        self.value_data = value
        self.enabled = None
        self.text = None
        self.slider_value = 0
        self.blocked = []

    def currentText(self):
        return self.value_data

    def value(self):
        return self.value_data

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setValue(self, value):
        self.slider_value = value

    def setText(self, text):
        self.text = text

    def blockSignals(self, blocked):
        self.blocked.append(bool(blocked))

    def sliderPosition(self):
        return self.slider_value


class FakeController:
    def __init__(self):
        self.running = False
        self.source_mode = "live"
        self.capture_settings = []
        self.start_count = 0
        self.stop_count = 0
        self.recording_result = None
        self.seek_calls = []

    def is_camera_running(self):
        return self.running

    def sync_capture_settings(self, *settings):
        self.capture_settings.append(settings)

    def start_camera(self):
        self.start_count += 1
        self.running = True

    def stop_camera(self):
        self.stop_count += 1
        self.running = False

    def toggle_recording(self):
        return self.recording_result

    def seek_playback(self, fraction):
        self.seek_calls.append(fraction)


class FakeProgress:
    def __init__(self):
        self.begin_count = 0

    def begin_drag(self):
        self.begin_count += 1

    def reset(self):
        return SimpleNamespace(
            enabled=False,
            update_slider=True,
            slider_value=0,
            label="--:-- / --:--",
        )

    def finish_drag(self, _position):
        return SimpleNamespace(
            fraction=0.25,
            view=SimpleNamespace(
                enabled=True,
                update_slider=True,
                slider_value=2500,
                label="00:01 / 00:04",
            ),
        )


class FakeWindow:
    def __init__(self):
        self.controller = FakeController()
        self.view_state = SimpleNamespace(
            set_camera_running=lambda: self.calls.append("camera_running"),
            set_camera_stopped=lambda: self.calls.append("camera_stopped"),
            set_recording_running=lambda: self.calls.append("recording_running"),
            set_recording_stopped=lambda **kwargs: self.calls.append(
                ("recording_stopped", kwargs)
            ),
        )
        self.calls = []
        self.palette_combo_box = FakeValueWidget("Dark")
        self.fps_spin_box = FakeValueWidget(60)
        self.replay_speed_combo_box = FakeValueWidget("2x")
        self.camera_image_label = FakeValueWidget()
        self.playback_progress_slider = FakeValueWidget()
        self.playback_time_label = FakeValueWidget()
        self.playback_progress = FakeProgress()
        self.predictions = SimpleNamespace(
            clear=lambda: self.calls.append("predictions_cleared")
        )


def _coordinator(window):
    return CameraUiCoordinator(
        CameraUiPorts(
            controller=window.controller,
            view_state=window.view_state,
            playback_progress=window.playback_progress,
            predictions=window.predictions,
            palette_combo_box=window.palette_combo_box,
            fps_spin_box=window.fps_spin_box,
            replay_speed_combo_box=window.replay_speed_combo_box,
            camera_image_label=window.camera_image_label,
            playback_progress_slider=window.playback_progress_slider,
            playback_time_label=window.playback_time_label,
            choose_input_file=lambda: None,
            process_events=lambda: None,
        )
    )


def test_camera_coordinator_starts_and_stops_through_one_entry_point():
    window = FakeWindow()
    coordinator = _coordinator(window)

    assert not hasattr(coordinator, "window")

    coordinator.toggle_camera()

    assert window.controller.start_count == 1
    assert window.controller.capture_settings == [("Dark", 60, 2.0)]
    assert "camera_running" in window.calls

    coordinator.toggle_camera()

    assert window.controller.stop_count == 1
    assert "camera_stopped" in window.calls
    assert window.camera_image_label.text == "相机未启动"


def test_camera_coordinator_applies_seek_view_and_forwards_fraction():
    window = FakeWindow()
    coordinator = _coordinator(window)

    coordinator.finish_progress_drag()

    assert window.playback_progress_slider.slider_value == 2500
    assert window.playback_time_label.text == "00:01 / 00:04"
    assert window.controller.seek_calls == [0.25]
    assert window.controller.capture_settings == [("Dark", 60, 2.0)]
