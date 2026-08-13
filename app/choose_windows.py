from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
)

from .display_names import (
    NOISE_FILTER_DISPLAY_NAMES,
    NOISE_FILTER_VALUES_BY_DISPLAY,
)


class ChooseWindow(QObject):
    """Own the settings controls embedded in the main-window sidebar."""

    settings_confirmed = pyqtSignal(object, str, str, int)

    def __init__(
        self,
        initial_mode="center",
        initial_roi=None,
        initial_noise_filter_type="none",
        initial_noise_filter_threshold_us=10000,
        parent=None,
    ):
        super().__init__(parent)
        self._dialog_parent = parent
        self._create_controls(parent)
        self.select_roi_button.clicked.connect(self.on_confirm)

        self.set_selected_mode(initial_mode, emit_signal=False)
        self.set_roi_values(initial_roi)
        self.set_noise_filter_values(
            initial_noise_filter_type,
            initial_noise_filter_threshold_us,
        )

    def _create_controls(self, parent):
        self.center_radioButton = QRadioButton("center", parent)
        self.center_radioButton.setObjectName("center_radioButton")
        self.center_radioButton.setMinimumSize(94, 34)

        self.eli_radioButton = QRadioButton("ellipse", parent)
        self.eli_radioButton.setObjectName("eli_radioButton")
        self.eli_radioButton.setMinimumSize(74, 34)

        for name in ("X", "Y", "Width", "Height"):
            editor = QLineEdit(parent)
            editor.setObjectName(f"{name}_edit")
            editor.setMinimumSize(180, 34)
            minimum = 0 if name in ("X", "Y") else 1
            editor.setValidator(QIntValidator(minimum, 2_147_483_647, editor))
            setattr(self, f"{name}_edit", editor)

        self.noise_filter_combo_box = QComboBox(parent)
        self.noise_filter_combo_box.setObjectName("noise_filter_combo_box")
        self.noise_filter_combo_box.setMinimumHeight(34)
        self.noise_filter_combo_box.addItems(NOISE_FILTER_DISPLAY_NAMES.values())

        self.noise_threshold_spin_box = QSpinBox(parent)
        self.noise_threshold_spin_box.setObjectName("noise_threshold_spin_box")
        self.noise_threshold_spin_box.setMinimumHeight(34)
        self.noise_threshold_spin_box.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.noise_threshold_spin_box.setRange(1, 1_000_000)
        self.noise_threshold_spin_box.setSingleStep(1000)

        self.select_roi_button = QPushButton("应用设置", parent)
        self.select_roi_button.setObjectName("select_roi_button")
        self.select_roi_button.setMinimumSize(0, 36)

    def _current_mode(self):
        if self.eli_radioButton.isChecked():
            return "ellipse"
        return "center"

    def _current_noise_filter(self):
        display = self.noise_filter_combo_box.currentText()
        return (
            NOISE_FILTER_VALUES_BY_DISPLAY.get(display, "none"),
            self.noise_threshold_spin_box.value(),
        )

    def set_selected_mode(self, mode, emit_signal=True):
        previous_center_block = self.center_radioButton.blockSignals(not emit_signal)
        previous_eli_block = self.eli_radioButton.blockSignals(not emit_signal)
        try:
            if mode == "ellipse":
                self.eli_radioButton.setChecked(True)
            else:
                self.center_radioButton.setChecked(True)
        finally:
            self.center_radioButton.blockSignals(previous_center_block)
            self.eli_radioButton.blockSignals(previous_eli_block)

    def set_roi_values(self, roi):
        if not roi:
            for editor in (self.X_edit, self.Y_edit, self.Width_edit, self.Height_edit):
                editor.clear()
            return
        x, y, width, height = roi
        self.X_edit.setText(str(x))
        self.Y_edit.setText(str(y))
        self.Width_edit.setText(str(width))
        self.Height_edit.setText(str(height))

    def set_noise_filter_values(self, filter_type, threshold_us):
        display = NOISE_FILTER_DISPLAY_NAMES.get(filter_type or "none", "None")
        index = self.noise_filter_combo_box.findText(display)
        if index >= 0:
            self.noise_filter_combo_box.setCurrentIndex(index)
        try:
            threshold_us = int(threshold_us)
        except (TypeError, ValueError):
            threshold_us = 10000
        self.noise_threshold_spin_box.setValue(max(1, threshold_us))

    def on_confirm(self):
        mode = self._current_mode()
        filter_type, threshold_us = self._current_noise_filter()
        roi_fields = (self.X_edit, self.Y_edit, self.Width_edit, self.Height_edit)
        has_roi_text = any(field.text().strip() for field in roi_fields)

        try:
            x = int(self.X_edit.text())
            y = int(self.Y_edit.text())
            width = int(self.Width_edit.text())
            height = int(self.Height_edit.text())
        except ValueError:
            if has_roi_text:
                QMessageBox.warning(
                    self._dialog_parent,
                    "区域无效",
                    "感兴趣区域未更新。请将所有输入框填写为整数，或全部留空。",
                )
                return
            self.settings_confirmed.emit(None, mode, filter_type, threshold_us)
            return

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            QMessageBox.warning(
                self._dialog_parent,
                "区域无效",
                "ROI 的 X、Y 必须大于等于 0，宽度和高度必须大于 0。",
            )
            return

        self.settings_confirmed.emit(
            (x, y, width, height),
            mode,
            filter_type,
            threshold_us,
        )
