"""Coordinate camera, recording, source selection, and playback UI actions."""

from dataclasses import dataclass
from typing import Any, Callable

from .view_state import source_is_file


SUPPORTED_PALETTES = {"Dark", "Light", "CoolWarm", "Gray"}
REPLAY_SPEEDS = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
}


@dataclass(frozen=True)
class CameraUiPorts:
    controller: Any
    view_state: Any
    playback_progress: Any
    predictions: Any
    palette_combo_box: Any
    fps_spin_box: Any
    replay_speed_combo_box: Any
    camera_image_label: Any
    playback_progress_slider: Any
    playback_time_label: Any
    choose_input_file: Callable[[], Any]
    process_events: Callable[[], None]


class CameraUiCoordinator:
    def __init__(self, ports):
        self.ports = ports

    def toggle_camera(self):
        ports = self.ports
        if not ports.controller.is_camera_running():
            self.sync_capture_settings()
            ports.controller.start_camera()
            ports.view_state.set_camera_running()
        else:
            self.stop_camera()

    def toggle_recording(self):
        ports = self.ports
        recording_started = ports.controller.toggle_recording()
        if recording_started is None:
            return
        if recording_started:
            ports.view_state.set_recording_running()
        else:
            ports.view_state.set_recording_stopped(enabled=True)

    def update_replay_speed(self):
        self.sync_capture_settings()
        self.ports.controller.update_replay_factor()

    def update_display_settings(self):
        self.sync_capture_settings()
        self.ports.controller.update_display_settings()

    def handle_playback_finished(self):
        self.stop_camera()

    def handle_playback_progress(self, current_us, total_us):
        ports = self.ports
        if not source_is_file(ports.controller):
            self.reset_playback_progress()
            return
        progress_view = ports.playback_progress.update(current_us, total_us)
        self.apply_playback_progress_view(progress_view)

    def stop_camera(self):
        ports = self.ports
        ports.controller.stop_camera()
        ports.view_state.set_camera_stopped()
        ports.view_state.set_recording_stopped(enabled=False)
        ports.camera_image_label.setText(
            "回放已停止" if source_is_file(ports.controller) else "相机未启动"
        )
        ports.predictions.clear()
        self.reset_playback_progress()

    def select_live_camera(self):
        ports = self.ports
        if not source_is_file(ports.controller):
            ports.view_state.set_live_camera()
            self.refresh_camera_view_state()
            return

        self.reset_playback_progress()
        ports.predictions.clear()
        ports.process_events()
        self.sync_capture_settings()
        ports.controller.set_live_camera()
        ports.view_state.set_live_camera()
        self.refresh_camera_view_state()

    def select_input_file(self):
        ports = self.ports
        file_path = ports.choose_input_file()
        if not file_path:
            return

        self.reset_playback_progress()
        ports.predictions.clear()
        ports.process_events()
        self.sync_capture_settings()
        ports.controller.set_input_file(file_path, restart_if_running=True)
        ports.view_state.set_input_file(file_path)
        self.refresh_camera_view_state()

    def refresh_camera_view_state(self):
        ports = self.ports
        if ports.controller.is_camera_running():
            ports.view_state.set_camera_running()
            return
        ports.view_state.set_camera_stopped()
        ports.view_state.set_recording_stopped(enabled=False)

    def selected_palette(self):
        selected = self.ports.palette_combo_box.currentText()
        if selected in SUPPORTED_PALETTES:
            return selected
        return "Dark"

    def sync_capture_settings(self):
        ports = self.ports
        ports.controller.sync_capture_settings(
            self.selected_palette(),
            ports.fps_spin_box.value(),
            self.selected_replay_factor(),
        )

    def selected_replay_factor(self):
        return REPLAY_SPEEDS.get(
            self.ports.replay_speed_combo_box.currentText(),
            1.0,
        )

    def begin_progress_drag(self):
        self.ports.playback_progress.begin_drag()

    def preview_progress_drag(self, value):
        progress_view = self.ports.playback_progress.preview(value)
        if progress_view is not None:
            self.apply_playback_progress_view(progress_view)

    def finish_progress_drag(self):
        ports = self.ports
        seek = ports.playback_progress.finish_drag(
            ports.playback_progress_slider.sliderPosition()
        )
        if seek is None:
            return
        self.apply_playback_progress_view(seek.view)
        self.sync_capture_settings()
        ports.predictions.clear()
        ports.controller.seek_playback(seek.fraction)

    def reset_playback_progress(self):
        self.apply_playback_progress_view(self.ports.playback_progress.reset())

    def apply_playback_progress_view(self, progress_view):
        ports = self.ports
        ports.playback_progress_slider.setEnabled(progress_view.enabled)
        if progress_view.update_slider:
            ports.playback_progress_slider.blockSignals(True)
            ports.playback_progress_slider.setValue(progress_view.slider_value)
            ports.playback_progress_slider.blockSignals(False)
        ports.playback_time_label.setText(progress_view.label)
