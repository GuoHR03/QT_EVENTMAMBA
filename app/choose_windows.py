import os
from PyQt6.QtWidgets import QWidget, QMessageBox
from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
base_dir = os.path.dirname(os.path.abspath(__file__))


class ChooseWindow(QWidget):
    roi_confirmed = pyqtSignal(int, int, int, int, str)

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(base_dir, "choose_form.ui")
        uic.loadUi(ui_path, self)
        self.select_roi_button.clicked.connect(self.on_confirm)

    def on_confirm(self):
        try:
            x = int(self.X_edit.text())
            y = int(self.Y_edit.text())
            width = int(self.Width_edit.text())
            height = int(self.Height_edit.text())

            if self.center_radioButton.isChecked():
                mode = "center"
            elif self.eli_radioButton.isChecked():
                mode = "ellipse"
            else:
                mode = "center"

            self.roi_confirmed.emit(x, y, width, height, mode)
            self.close()
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", "请输入有效的整数")
