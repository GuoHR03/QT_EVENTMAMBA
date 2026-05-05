import os
from PyQt6.QtWidgets import QWidget
from PyQt6 import uic
base_dir = os.path.dirname(os.path.abspath(__file__))


class ChooseWindow(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(base_dir, "choose_form.ui")
        uic.loadUi(ui_path, self)
