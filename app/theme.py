APP_STYLE = """
QWidget {
    background: #f3f5f8;
    color: #18202b;
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    font-size: 13px;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #d9e0ea;
    border-radius: 8px;
    margin-top: 16px;
    padding: 0;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #39465a;
}

QLabel#camera_image_label {
    background: #080b10;
    color: #7b8794;
    border: 1px solid #111827;
    border-radius: 8px;
    font-size: 18px;
}

QLabel#weight_path_label,
QLabel#input_file_label {
    background: #f8fafc;
    border: 1px solid #d8e0ea;
    border-radius: 6px;
    color: #526174;
    padding: 0 10px;
}

QTextEdit#log_text_edit {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #dbeafe;
    font-family: Consolas, "Microsoft YaHei", monospace;
    font-size: 12px;
    padding: 8px;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}

QPushButton:hover {
    background: #f8fafc;
    border-color: #9fb1c7;
}

QPushButton:pressed {
    background: #edf2f7;
}

QPushButton:disabled {
    background: #e8edf3;
    color: #94a3b8;
    border-color: #d7dee8;
}

QPushButton#start_camera_button,
QPushButton#load_model_button,
QPushButton#select_roi_button {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#start_camera_button:hover,
QPushButton#load_model_button:hover,
QPushButton#select_roi_button:hover {
    background: #1d4ed8;
}

QPushButton#unload_model_button,
QPushButton#record_button {
    border-color: #f3b4b4;
    color: #b42318;
}

QComboBox,
QDoubleSpinBox,
QSpinBox,
QLineEdit {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 3px 8px;
    min-height: 24px;
}

QComboBox:focus,
QDoubleSpinBox:focus,
QSpinBox:focus,
QLineEdit:focus {
    border-color: #2563eb;
}

QLabel#playback_time_label {
    background: transparent;
    border: none;
    color: #526174;
    font-family: Consolas, "Microsoft YaHei", monospace;
    font-size: 12px;
}

QSlider#playback_progress_slider::groove:horizontal {
    background: #d8e0ea;
    border-radius: 3px;
    height: 6px;
}

QSlider#playback_progress_slider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}

QSlider#playback_progress_slider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    height: 14px;
    width: 14px;
    margin: -5px 0;
}

QSlider#playback_progress_slider:disabled::sub-page:horizontal {
    background: #94a3b8;
}

QWidget#ChooseWindowForm QGroupBox {
    background: #ffffff;
}

QWidget#ChooseWindowForm QFrame#mode_card_frame,
QWidget#ChooseWindowForm QFrame#roi_card_frame,
QWidget#ChooseWindowForm QFrame#noise_filter_widget {
    background: #ffffff;
    border: 1px solid #d9e0ea;
    border-radius: 8px;
}

QWidget#ChooseWindowForm QLabel#mode_title_label,
QWidget#ChooseWindowForm QLabel#roi_title_label,
QWidget#ChooseWindowForm QLabel#denoise_title_label {
    background: transparent;
    border: none;
    color: #1f2a44;
    font-size: 15px;
    font-weight: 600;
    padding-left: 4px;
    min-height: 24px;
}

QWidget#ChooseWindowForm QLabel {
    background: transparent;
    border: none;
}

QWidget#ChooseWindowForm QRadioButton {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 5px 10px;
}

QWidget#ChooseWindowForm QRadioButton:checked {
    background: #eff6ff;
    border-color: #93c5fd;
    color: #1d4ed8;
    font-weight: 600;
}

QWidget#ChooseWindowForm QComboBox#noise_filter_combo_box,
QWidget#ChooseWindowForm QSpinBox#noise_threshold_spin_box {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 28px;
}

QWidget#ChooseWindowForm QComboBox#noise_filter_combo_box::drop-down {
    border-left: none;
    width: 30px;
}

"""


def apply_app_theme(widget):
    widget.setStyleSheet(APP_STYLE)
