import os
from PyQt6.QtWidgets import QWidget
from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
base_dir = os.path.dirname(os.path.abspath(__file__))


class choose_Window(QWidget):
    roi_applied = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(base_dir, "choose_form.ui")
        uic.loadUi(ui_path, self)
        self.applyROI_btn.clicked.connect(self.ROI_apply_clicked)

    def ROI_apply_clicked(self):
        roi_x = int(self.x_Edit.text())
        roi_y = int(self.y_Edit.text())
        roi_width = int(self.width_Edit.text())
        roi_height = int(self.height_Edit.text())
        self.roi_applied.emit(roi_x, roi_y, roi_width, roi_height)
