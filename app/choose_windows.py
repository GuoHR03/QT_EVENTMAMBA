from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

try:
    from .bootstrap import app_resource_path
    from .theme import apply_app_theme
except ImportError:
    from bootstrap import app_resource_path
    from theme import apply_app_theme

FILTER_DISPLAY_TO_VALUE = {
    "None": "none",
    "Activity": "activity",
    "Trail": "trail",
    "STC": "stc",
    "AntiFlicker": "anti_flicker",
}
FILTER_VALUE_TO_DISPLAY = {value: display for display, value in FILTER_DISPLAY_TO_VALUE.items()}


class ChooseWindow(QWidget):
    settings_confirmed = pyqtSignal(object, str, str, int)

    def __init__(
        self,
        initial_mode="center",
        initial_roi=None,
        initial_noise_filter_type="none",
        initial_noise_filter_threshold_us=10000,
        parent=None,
        embedded=False,
    ):
        super().__init__(parent)
        self._embedded = embedded
        ui_path = app_resource_path("choose_form.ui")
        uic.loadUi(ui_path, self)
        if embedded:
            self._configure_embedded_layout()
        apply_app_theme(self)
        self.select_roi_button.clicked.connect(self.on_confirm)

        self.set_selected_mode(initial_mode, emit_signal=False)
        self.set_roi_values(initial_roi)
        self.set_noise_filter_values(initial_noise_filter_type, initial_noise_filter_threshold_us)

    def _configure_embedded_layout(self):
        """Adapt the dialog-oriented form to the narrow main-window sidebar."""
        self.setMinimumSize(0, 0)
        self.choose_main_layout.setContentsMargins(12, 12, 12, 12)
        self.choose_main_layout.setSpacing(6)
        self.mode_card_frame.setMinimumHeight(0)
        self.roi_card_frame.setMinimumHeight(0)
        self.noise_filter_widget.setMinimumHeight(0)
        self.mode_layout.setContentsMargins(8, 6, 8, 6)
        self.mode_layout.setSpacing(8)

        while self.roi_grid_layout.count():
            self.roi_grid_layout.takeAt(0)
        self.roi_grid_layout.setContentsMargins(8, 8, 8, 8)
        self.roi_grid_layout.setHorizontalSpacing(8)
        self.roi_grid_layout.setVerticalSpacing(6)
        roi_rows = (
            (self.X_label, self.X_edit),
            (self.Y_label, self.Y_edit),
            (self.Width_label, self.Width_edit),
            (self.Height_label, self.Height_edit),
        )
        for row, (label, editor) in enumerate(roi_rows):
            label.setMinimumWidth(54)
            editor.setMinimumWidth(0)
            self.roi_grid_layout.addWidget(label, row, 0)
            self.roi_grid_layout.addWidget(editor, row, 1)
        self.roi_grid_layout.setColumnStretch(1, 1)
        roi_height = (
            sum(editor.minimumHeight() for _, editor in roi_rows)
            + self.roi_grid_layout.verticalSpacing() * (len(roi_rows) - 1)
            + 16
        )
        self.roi_card_frame.setMinimumHeight(roi_height)

        self.noise_filter_formLayout.setContentsMargins(8, 8, 8, 8)
        self.noise_filter_formLayout.setHorizontalSpacing(8)
        self.noise_filter_formLayout.setVerticalSpacing(6)
        self.noise_filter_label.setMinimumWidth(88)
        self.noise_threshold_label.setMinimumWidth(88)
        noise_height = (
            self.noise_filter_combo_box.minimumHeight()
            + self.noise_threshold_spin_box.minimumHeight()
            + self.noise_filter_formLayout.verticalSpacing()
            + 16
        )
        self.noise_filter_widget.setMinimumHeight(noise_height)
        self.select_roi_button.setText("应用设置")
        self.select_roi_button.setMinimumSize(0, 36)

    def _current_mode(self):
        if self.eli_radioButton.isChecked():
            return "ellipse"
        return "center"

    def _current_noise_filter(self):
        display = self.noise_filter_combo_box.currentText()
        return FILTER_DISPLAY_TO_VALUE.get(display, "none"), self.noise_threshold_spin_box.value()

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
            return
        x, y, width, height = roi
        self.X_edit.setText(str(x))
        self.Y_edit.setText(str(y))
        self.Width_edit.setText(str(width))
        self.Height_edit.setText(str(height))

    def set_noise_filter_values(self, filter_type, threshold_us):
        display = FILTER_VALUE_TO_DISPLAY.get(filter_type or "none", "None")
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
                QMessageBox.warning(self, "区域无效", "感兴趣区域未更新。请将所有输入框填写为整数，或全部留空。")
                return
            self.settings_confirmed.emit(None, mode, filter_type, threshold_us)
            if not self._embedded:
                self.close()
            return

        self.settings_confirmed.emit((x, y, width, height), mode, filter_type, threshold_us)
        if not self._embedded:
            self.close()
