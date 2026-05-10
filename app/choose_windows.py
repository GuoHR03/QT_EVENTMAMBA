import os

from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

base_dir = os.path.dirname(os.path.abspath(__file__))


class ChooseWindow(QWidget):
    roi_confirmed = pyqtSignal(int, int, int, int, str)
    mode_confirmed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(base_dir, "choose_form.ui")
        uic.loadUi(ui_path, self)
        self.select_roi_button.clicked.connect(self.on_confirm)

        self.center_radioButton.toggled.connect(self._on_mode_toggled)
        self.eli_radioButton.toggled.connect(self._on_mode_toggled)

        # Default to center and broadcast it immediately when the window is shown.
        self.center_radioButton.setChecked(True)

    def _current_mode(self):
        if self.eli_radioButton.isChecked():
            return "ellipse"
        return "center"

    def selected_mode(self):
        return self._current_mode()

    def _on_mode_toggled(self, checked):
        if checked:
            self.mode_confirmed.emit(self._current_mode())

    def on_confirm(self):
        mode = self._current_mode()

        try:
            x = int(self.X_edit.text())
            y = int(self.Y_edit.text())
            width = int(self.Width_edit.text())
            height = int(self.Height_edit.text())
        except ValueError:
            if any(field.text().strip() for field in (self.X_edit, self.Y_edit, self.Width_edit, self.Height_edit)):
                QMessageBox.warning(self, "Invalid ROI", "ROI was not updated. Fill all ROI fields with integers, or leave them blank.")
                return
            self.close()
            return

        self.roi_confirmed.emit(x, y, width, height, mode)
        self.close()
