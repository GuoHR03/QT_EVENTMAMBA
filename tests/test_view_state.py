from types import SimpleNamespace

from app.view_state import MainViewState, source_is_file


class FakeStyle:
    def unpolish(self, _widget):
        pass

    def polish(self, _widget):
        pass


class FakeButton:
    def __init__(self):
        self.text = ""
        self.enabled = True
        self.properties = {}
        self._style = FakeStyle()

    def setText(self, text):
        self.text = text

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setProperty(self, name, value):
        self.properties[name] = value

    def style(self):
        return self._style


class FakeLabel:
    def __init__(self):
        self.text = ""
        self.tooltip = ""

    def setText(self, text):
        self.text = text

    def setToolTip(self, tooltip):
        self.tooltip = tooltip


class FakeView:
    def __init__(self):
        self.start_camera_button = FakeButton()
        self.record_button = FakeButton()
        self.load_model_button = FakeButton()
        self.unload_model_button = FakeButton()
        self.restart_model_button = FakeButton()
        self.select_weight_button = FakeButton()
        self.input_file_label = FakeLabel()
        self.weight_path_label = FakeLabel()
        self.statuses = []
        self.source_paths = []
        self.controller = SimpleNamespace(source_mode="live", input_file_path=None)

    def set_runtime_status(self, target, text, state):
        self.statuses.append((target, text, state))

    def set_source_status(self, file_path):
        self.source_paths.append(file_path)


def test_view_state_updates_camera_and_recording_visual_roles():
    view = FakeView()
    state = MainViewState(view)

    state.set_camera_running()
    state.set_recording_running()

    assert view.start_camera_button.text == "停止相机"
    assert view.start_camera_button.properties["buttonRole"] == "primary"
    assert view.record_button.text == "停止录制"
    assert view.record_button.properties["buttonRole"] == "danger"
    assert ("camera", "运行中", "active") in view.statuses
    assert ("camera", "录制中", "danger") in view.statuses

    state.set_camera_stopped()
    state.set_recording_stopped(enabled=False)

    assert view.start_camera_button.text == "启动相机"
    assert view.record_button.enabled is False
    assert view.record_button.properties["buttonRole"] == "dangerOutline"
    assert ("camera", "已停止", "idle") in view.statuses


def test_file_mode_uses_playback_labels_and_disables_recording():
    view = FakeView()
    view.controller.source_mode = "raw"
    view.controller.input_file_path = "C:/records/events.raw"
    state = MainViewState(view)

    state.set_camera_running()

    assert view.start_camera_button.text == "停止播放"
    assert view.record_button.enabled is False


def test_view_state_updates_model_status_and_selected_paths():
    view = FakeView()
    state = MainViewState(view)

    state.set_model_starting()
    state.set_model_running()
    state.set_input_file("C:/records/events.raw")
    state.set_weight_file("C:/models/model.pth")

    assert ("model", "推理启动中", "pending") in view.statuses
    assert ("model", "推理运行中", "active") in view.statuses
    assert view.restart_model_button.enabled is True
    assert view.select_weight_button.enabled is False
    assert view.input_file_label.text == "events.raw"
    assert view.weight_path_label.text == "model.pth"
    assert view.source_paths == ["C:/records/events.raw"]

    state.set_model_error()
    assert view.load_model_button.text == "重试启动"
    assert view.unload_model_button.enabled is True
    assert view.restart_model_button.enabled is True
    assert ("model", "推理错误", "danger") in view.statuses

    state.set_live_camera()
    assert view.input_file_label.text == "实时相机"
    assert view.source_paths[-1] is None


def test_view_state_disables_model_controls_while_stopping():
    view = FakeView()
    state = MainViewState(view)

    state.set_model_stopping()

    assert view.unload_model_button.text == "停止中..."
    assert not view.load_model_button.enabled
    assert not view.unload_model_button.enabled
    assert not view.restart_model_button.enabled
    assert ("model", "推理停止中", "pending") in view.statuses


def test_explicit_source_mode_takes_priority_over_legacy_path():
    controller = SimpleNamespace(source_mode="live", input_file_path="stale.raw")
    assert not source_is_file(controller)

    controller.source_mode = "raw"
    controller.input_file_path = None
    assert source_is_file(controller)
