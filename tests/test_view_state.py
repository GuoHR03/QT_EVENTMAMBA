from app.view_state import MainViewState


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
        self.select_weight_button = FakeButton()
        self.input_file_label = FakeLabel()
        self.weight_path_label = FakeLabel()
        self.statuses = []
        self.source_paths = []

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


def test_view_state_updates_model_status_and_selected_paths():
    view = FakeView()
    state = MainViewState(view)

    state.set_model_loading()
    state.set_model_loaded()
    state.set_input_file("C:/records/events.aedat4")
    state.set_weight_file("C:/models/model.pth")

    assert ("model", "模型加载中", "pending") in view.statuses
    assert ("model", "模型已加载", "active") in view.statuses
    assert view.input_file_label.text == "events.aedat4"
    assert view.weight_path_label.text == "model.pth"
    assert view.source_paths == ["C:/records/events.aedat4"]
