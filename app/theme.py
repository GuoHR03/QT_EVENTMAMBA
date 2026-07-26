APP_STYLE = """
QWidget {
    background: #f4f6f9;
    color: #18202b;
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    font-size: 13px;
}

QToolTip {
    background-color: #f8fafc;
    color: #334155;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px 9px;
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    font-size: 12px;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #dce3ec;
    border-radius: 7px;
    margin-top: 14px;
    padding: 0;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #334155;
    font-size: 12px;
}

QWidget#viewer_widget,
QWidget#viewer_header_widget,
QWidget#camera_viewport_widget,
QWidget#playback_progress_widget,
QWidget#log_header_widget,
QWidget#control_panel_widget {
    background: transparent;
}

QScrollArea#control_panel_scroll_area {
    background: #ffffff;
    border: 1px solid #d7dce3;
    border-radius: 7px;
}

QScrollArea#control_panel_scroll_area > QWidget > QWidget {
    background: #ffffff;
    border: none;
}

QScrollArea#control_panel_scroll_area QWidget#control_panel_widget {
    min-width: 0;
}

QScrollArea#control_panel_scroll_area QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}

QScrollArea#control_panel_scroll_area QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 28px;
}

QScrollArea#control_panel_scroll_area QScrollBar::add-line:vertical,
QScrollArea#control_panel_scroll_area QScrollBar::sub-line:vertical,
QScrollArea#control_panel_scroll_area QScrollBar::add-page:vertical,
QScrollArea#control_panel_scroll_area QScrollBar::sub-page:vertical {
    background: transparent;
    border: none;
    height: 0;
}

QFrame[uiRole="accordionSection"] {
    background: #ffffff;
    border: 1px solid #d7dce3;
    border-bottom: none;
    border-radius: 0;
}

QFrame[uiRole="accordionSection"]:first-child {
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}

QPushButton[uiRole="accordionHeader"] {
    background: #fafafa;
    color: #3b78df;
    border: none;
    border-bottom: 1px solid #d7dce3;
    border-radius: 0;
    padding: 0 18px;
    min-height: 54px;
    text-align: left;
    font-size: 15px;
    font-weight: 400;
}

QPushButton[uiRole="accordionHeader"]:hover {
    background: #f4f7fb;
    border: none;
    border-bottom: 1px solid #cbd3dd;
}

QPushButton[uiRole="accordionHeader"]:checked {
    background: #fafafa;
    color: #2563c7;
    border: none;
    border-bottom: 1px solid #d7dce3;
}

QPushButton[uiRole="accordionHeader"]:disabled {
    background: #fafafa;
    color: #a6afbc;
    border: none;
    border-bottom: 1px solid #d7dce3;
}

QGroupBox[uiRole="accordionContent"] {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #d7dce3;
    border-radius: 0;
    margin-top: 0;
}

QGroupBox[uiRole="controlSubsection"] {
    background: #f8fafc;
    border: 1px solid #dce3ec;
    border-radius: 6px;
    margin-top: 18px;
}

QGroupBox[uiRole="controlSubsection"]::title {
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 5px;
    color: #526174;
    font-size: 12px;
    font-weight: 600;
}

QLabel#viewer_title_label,
QLabel#log_title_label {
    background: transparent;
    color: #253247;
    border: none;
    font-size: 13px;
    font-weight: 600;
}

QLabel[uiRole="fieldTitle"] {
    background: transparent;
    border: none;
    color: #526174;
    font-size: 12px;
    padding-top: 4px;
}

QLabel[uiRole="statusChip"] {
    background: #edf1f5;
    color: #64748b;
    border: 1px solid #dce3ea;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    min-height: 18px;
}

QLabel[uiRole="statusChip"][statusState="active"] {
    background: #e9f8ef;
    color: #197044;
    border-color: #b9e5ca;
}

QLabel[uiRole="statusChip"][statusState="pending"] {
    background: #fff7e6;
    color: #9a6700;
    border-color: #f2d49b;
}

QLabel[uiRole="statusChip"][statusState="danger"] {
    background: #fff0f0;
    color: #b42318;
    border-color: #efb0b0;
}

QLabel[uiRole="statusChip"][statusState="info"] {
    background: #edf5ff;
    color: #245ea8;
    border-color: #c5daf5;
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
    min-height: 26px;
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

QPushButton#settings_panel_button {
    background: #4383ce;
    border: 1px solid #4383ce;
    border-radius: 7px;
    padding: 0;
    min-width: 40px;
    max-width: 42px;
    min-height: 40px;
    max-height: 42px;
}

QPushButton#settings_panel_button:hover {
    background: #3474bf;
    border-color: #3474bf;
}

QPushButton#settings_panel_button:pressed,
QPushButton#settings_panel_button:checked {
    background: #245ea8;
    border-color: #245ea8;
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

QPushButton[buttonRole="primary"] {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}

QPushButton[buttonRole="primary"]:hover {
    background: #1d4ed8;
}

QPushButton[buttonRole="danger"] {
    background: #dc2626;
    border-color: #b91c1c;
    color: #ffffff;
    font-weight: 600;
}

QPushButton[buttonRole="danger"]:hover {
    background: #b91c1c;
}

QPushButton[buttonRole="dangerOutline"] {
    background: #ffffff;
    border-color: #efb0b0;
    color: #b42318;
}

QPushButton#clear_log_button,
QPushButton#log_toggle_button {
    background: transparent;
    border: 1px solid transparent;
    color: #526174;
    padding: 2px 8px;
    min-height: 20px;
}

QPushButton#clear_log_button:hover,
QPushButton#log_toggle_button:hover {
    background: #edf2f7;
    border-color: #d8e0ea;
}

QComboBox,
QDoubleSpinBox,
QSpinBox,
QLineEdit {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 3px 8px;
    min-height: 26px;
}

QComboBox:hover,
QDoubleSpinBox:hover,
QSpinBox:hover,
QLineEdit:hover {
    border-color: #aab8c8;
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

QWidget#control_panel_widget QLabel#palette_text_label,
QWidget#control_panel_widget QLabel#speed_text_label,
QWidget#control_panel_widget QLabel#fps_text_label {
    background: transparent;
    color: #526174;
    border: none;
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

QGroupBox[uiRole="accordionContent"] QWidget#ChooseWindowForm {
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
