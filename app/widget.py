from PyQt6 import uic
from PyQt6.QtCore import QSize, QTimer, Qt
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from backend.event_processing import normalize_roi

from .bootstrap import app_resource_path
from .choose_windows import ChooseWindow
from .controller import AppController
from .file_dialogs import choose_input_file, choose_weights_file
from .inference_operation import InferenceOperationThread
from .log_formatter import (
    backend_message,
    mode_display_name,
    noise_settings_message,
    roi_settings_message,
)
from .prediction_overlay import draw_prediction
from .prediction_state import PredictionState
from .settings import AppSettings
from .theme import apply_app_theme
from .ui_log import log_level_for_message
from .ui_status import source_display_name
from .view_state import MainViewState, source_is_file

SUPPORTED_PALETTES = {"Dark", "Light", "CoolWarm", "Gray"}
REPLAY_SPEEDS = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
}
PLAYBACK_SLIDER_MAX = 10000
INFERENCE_START = "start"
INFERENCE_STOP = "stop"
INFERENCE_RESTART = "restart"
INFERENCE_CLEANUP = "cleanup"
INFERENCE_CLOSE = "close"


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(app_resource_path("form.ui"), self)
        self.settings = AppSettings()
        self._init_workspace_ui()
        apply_app_theme(self)
        self._set_initial_window_geometry()

        self.controller = AppController(self.settings)
        self._configure_inference_runtime_ui()
        self.view_state = MainViewState(self)
        self.predictions = PredictionState(interval_ms=20)
        self._is_dragging_progress = False
        self._progress_total_us = 0
        self._last_frame_size = None
        self._inference_operation = None
        self._close_pending = False
        self._close_ready = False
        self._cleanup_after_operation = False
        self._last_inference_state = None

        self._connect_signals()
        self._init_view_state()
        self._inference_health_timer = QTimer(self)
        self._inference_health_timer.setInterval(1000)
        self._inference_health_timer.timeout.connect(self._refresh_inference_state)
        self._inference_health_timer.start()

    def _configure_inference_runtime_ui(self):
        runtime_name = self.controller.inference_runtime_display_name
        self.runtime_name_label.setText(f"推理后端：{runtime_name}")
        self.runtime_name_label.setToolTip(
            "推理后端由运行环境配置，界面中不可切换"
        )
        if self.controller.inference_runtime_kind != "windows":
            return

        self.select_weight_button.setText("选择 ONNX 模型")
        self.select_weight_button.setToolTip("选择已转换的 Windows ONNX 模型")
        self.weight_path_label.setText("尚未选择 ONNX 模型")
        self.roi_settings_editor.eli_radioButton.setToolTip(
            "使用 Windows ONNX/CUDA 椭圆模型输出位置、长短轴和角度"
        )

    def _set_initial_window_geometry(self):
        """Choose a compact 3:2 workspace instead of mirroring a wide screen."""
        self.setMinimumSize(900, 620)
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1050, 700)
            return

        available = screen.availableGeometry()
        height = min(760, max(650, int(available.height() * 0.72)))
        # Derive width from height so a 16:9 monitor does not produce an
        # unnecessarily wide application window. The 3:2 shell still leaves
        # enough room for the 250 px settings panel when it is opened.
        width = min(1140, max(975, int(height * 1.5)))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def _init_workspace_ui(self):
        self._init_playback_progress_ui()
        self._init_control_panel_ui()
        self._init_log_panel_ui()

    def _init_playback_progress_ui(self):
        self.playback_progress_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.playback_progress_slider.setObjectName("playback_progress_slider")
        self.playback_progress_slider.setRange(0, PLAYBACK_SLIDER_MAX)
        self.playback_progress_slider.setTracking(False)
        self.playback_progress_slider.setEnabled(False)

        self.playback_time_label = QLabel("--:-- / --:--", self)
        self.playback_time_label.setObjectName("playback_time_label")
        self.playback_time_label.setMinimumWidth(138)
        self.playback_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        image_item = self.content_horizontal_layout.takeAt(0)
        image_widget = image_item.widget()
        viewer_widget = QWidget(self)
        viewer_widget.setObjectName("viewer_widget")
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)

        viewer_header = QWidget(viewer_widget)
        viewer_header.setObjectName("viewer_header_widget")
        self.viewer_header_widget = viewer_header
        viewer_header_layout = QHBoxLayout(viewer_header)
        viewer_header_layout.setContentsMargins(2, 0, 2, 0)
        viewer_header_layout.setSpacing(8)

        viewer_title = QLabel("事件画面", viewer_header)
        viewer_title.setObjectName("viewer_title_label")
        viewer_header_layout.addWidget(viewer_title)
        viewer_header_layout.addStretch(1)

        self.source_status_label = self._create_status_chip("实时输入", "info", viewer_header)
        self.camera_status_label = self._create_status_chip("已停止", "idle", viewer_header)
        self.model_status_label = self._create_status_chip("模型未加载", "idle", viewer_header)
        self.mode_status_label = self._create_status_chip("中心点", "info", viewer_header)
        viewer_header_layout.addWidget(self.source_status_label)
        viewer_header_layout.addWidget(self.camera_status_label)
        viewer_header_layout.addWidget(self.model_status_label)
        viewer_header_layout.addWidget(self.mode_status_label)

        viewer_layout.addWidget(
            viewer_header,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        self.camera_viewport_widget = QWidget(viewer_widget)
        self.camera_viewport_widget.setObjectName("camera_viewport_widget")
        camera_viewport_layout = QVBoxLayout(self.camera_viewport_widget)
        camera_viewport_layout.setContentsMargins(0, 0, 0, 0)
        camera_viewport_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_widget.setMinimumSize(0, 0)
        image_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        camera_viewport_layout.addWidget(image_widget)
        viewer_layout.addWidget(self.camera_viewport_widget, 1)

        self.playback_progress_widget = QWidget(viewer_widget)
        self.playback_progress_widget.setObjectName("playback_progress_widget")
        progress_layout = QHBoxLayout(self.playback_progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        progress_layout.addWidget(self.playback_progress_slider, 1)
        progress_layout.addWidget(self.playback_time_label)

        self.settings_panel_button = QPushButton(self.playback_progress_widget)
        self.settings_panel_button.setObjectName("settings_panel_button")
        self.settings_panel_button.setCheckable(True)
        self.settings_panel_button.setFixedSize(42, 42)
        self.settings_panel_button.setIcon(self._create_settings_panel_icon())
        self.settings_panel_button.setIconSize(QSize(22, 22))
        self.settings_panel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_panel_button.setToolTip("展开右侧设置面板")
        self.settings_panel_button.setAccessibleName("显示或隐藏右侧设置面板")
        progress_layout.addWidget(self.settings_panel_button)
        viewer_layout.addWidget(
            self.playback_progress_widget,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )

        self.content_horizontal_layout.insertWidget(0, viewer_widget, 1)
        self.content_horizontal_layout.setStretch(0, 1)
        self.content_horizontal_layout.setStretch(1, 0)

    def _init_control_panel_ui(self):
        self.setWindowTitle("事件相机推理工具")
        self.control_panel_widget.setMinimumWidth(0)
        self.control_panel_widget.setMaximumWidth(310)
        self.control_panel_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        # Height is updated explicitly when accordion sections change. Keep
        # horizontal sizing flexible so long file names cannot widen the
        # scroll area's content beyond its viewport.
        self.control_panel_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        panel_index = self.content_horizontal_layout.indexOf(self.control_panel_widget)
        self.content_horizontal_layout.removeWidget(self.control_panel_widget)
        self.control_panel_scroll_area = QScrollArea(self)
        self.control_panel_scroll_area.setObjectName("control_panel_scroll_area")
        self.control_panel_scroll_area.setMinimumWidth(250)
        self.control_panel_scroll_area.setMaximumWidth(310)
        self.control_panel_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self.control_panel_scroll_area.setWidgetResizable(True)
        self.control_panel_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.control_panel_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.control_panel_scroll_area.setWidget(self.control_panel_widget)
        self.content_horizontal_layout.insertWidget(panel_index, self.control_panel_scroll_area)

        self.palette_text_label.setText("配色")
        self.speed_text_label.setText("回放速度")
        self.fps_text_label.setText("帧率")
        self.select_weight_button.setText("选择权重")
        self.load_model_button.setText("启动推理")
        self.unload_model_button.setText("停止推理")
        self.restart_model_button.setText("重启推理")
        self.live_camera_button.setText("实时相机")
        self.select_input_file_button.setText("选择 RAW 文件")
        self.weight_path_label.setText("尚未选择权重")
        self.input_file_label.setText("实时相机")

        self.palette_combo_box.setToolTip("选择事件极性的显示配色")
        self.replay_speed_combo_box.setToolTip("调整离线文件的回放速度")
        self.live_camera_button.setToolTip("切换到已连接的实时事件相机")
        self.select_input_file_button.setToolTip(
            "选择 RAW 事件文件；H5 和 AEDAT4 作为兼容格式保留"
        )
        self.fps_spin_box.setToolTip(
            "控制画面帧率和每帧事件累计时间，不影响模型的 20 ms 推理窗口"
        )
        self.record_button.setText("录制 RAW")
        self.record_button.setToolTip("仅实时相机支持录制 RAW 数据")

        self.roi_settings_editor = ChooseWindow(
            initial_mode=self.settings.prediction_mode,
            initial_roi=self.settings.roi,
            initial_noise_filter_type=self.settings.noise_filter_type,
            initial_noise_filter_threshold_us=self.settings.noise_filter_threshold_us,
            parent=self.control_panel_widget,
        )

        self._build_logical_control_groups()

        self._init_control_panel_accordion()
        self.control_panel_scroll_area.setVisible(False)

    def _build_logical_control_groups(self):
        """Regroup existing controls by workflow without replacing their signals."""
        old_groups = (
            self.model_group_box,
            self.input_group_box,
            self.settings_group_box,
            self.capture_group_box,
        )
        for group in old_groups:
            self.control_panel_layout.removeWidget(group)
            group.hide()

        flexible_controls = (
            self.input_file_label,
            self.live_camera_button,
            self.select_input_file_button,
            self.replay_speed_combo_box,
            self.start_camera_button,
            self.record_button,
            self.weight_path_label,
            self.runtime_name_label,
            self.select_weight_button,
            self.load_model_button,
            self.unload_model_button,
            self.restart_model_button,
            self.palette_combo_box,
            self.fps_spin_box,
            self.roi_settings_editor.noise_filter_combo_box,
            self.roi_settings_editor.noise_threshold_spin_box,
            self.roi_settings_editor.X_edit,
            self.roi_settings_editor.Y_edit,
            self.roi_settings_editor.Width_edit,
            self.roi_settings_editor.Height_edit,
        )
        for control in flexible_controls:
            control.setMinimumWidth(0)
            control.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                control.sizePolicy().verticalPolicy(),
            )
        self.input_file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )

        self.source_group_box = QGroupBox(self.control_panel_widget)
        source_layout = QVBoxLayout(self.source_group_box)
        source_layout.setContentsMargins(12, 12, 12, 12)
        source_layout.setSpacing(8)
        source_mode_layout = QHBoxLayout()
        source_mode_layout.setSpacing(8)
        source_mode_layout.addWidget(self.live_camera_button)
        source_mode_layout.addWidget(self.select_input_file_button)
        source_layout.addLayout(source_mode_layout)
        source_layout.addWidget(self.input_file_label)

        self.playback_group_box = QGroupBox(self.control_panel_widget)
        playback_layout = QGridLayout(self.playback_group_box)
        playback_layout.setContentsMargins(12, 12, 12, 12)
        playback_layout.setHorizontalSpacing(10)
        playback_layout.setVerticalSpacing(8)
        playback_layout.addWidget(self.speed_text_label, 0, 0)
        playback_layout.addWidget(self.replay_speed_combo_box, 0, 1)
        playback_layout.addWidget(self.fps_text_label, 1, 0)
        playback_layout.addWidget(self.fps_spin_box, 1, 1)
        playback_layout.addWidget(self.start_camera_button, 2, 0, 1, 2)
        playback_layout.setColumnStretch(1, 1)

        self.recording_group_box = QGroupBox(self.control_panel_widget)
        recording_layout = QVBoxLayout(self.recording_group_box)
        recording_layout.setContentsMargins(12, 12, 12, 12)
        recording_layout.addWidget(self.record_button)

        self.inference_group_box = QGroupBox(self.control_panel_widget)
        inference_layout = QVBoxLayout(self.inference_group_box)
        inference_layout.setContentsMargins(12, 12, 12, 12)
        inference_layout.setSpacing(8)
        inference_layout.addWidget(self.runtime_name_label)
        inference_layout.addWidget(self.weight_path_label)
        inference_layout.addWidget(self.select_weight_button)
        model_buttons = QHBoxLayout()
        model_buttons.setSpacing(8)
        model_buttons.addWidget(self.load_model_button)
        model_buttons.addWidget(self.unload_model_button)
        model_buttons.addWidget(self.restart_model_button)
        inference_layout.addLayout(model_buttons)
        self.prediction_mode_group_box = QGroupBox(self.control_panel_widget)
        mode_layout = QHBoxLayout(self.prediction_mode_group_box)
        mode_layout.setContentsMargins(12, 12, 12, 12)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.roi_settings_editor.center_radioButton)
        mode_layout.addWidget(self.roi_settings_editor.eli_radioButton)
        mode_layout.addStretch(1)

        self.model_prediction_group_box = QGroupBox(self.control_panel_widget)
        model_prediction_layout = QVBoxLayout(self.model_prediction_group_box)
        model_prediction_layout.setContentsMargins(10, 8, 10, 10)
        model_prediction_layout.setSpacing(8)
        for group_box, title in (
            (self.prediction_mode_group_box, "预测模式"),
            (self.inference_group_box, "模型管理"),
        ):
            group_box.setTitle(title)
            group_box.setProperty("uiRole", "controlSubsection")
            model_prediction_layout.addWidget(group_box)

        self.processing_group_box = QGroupBox(self.control_panel_widget)
        processing_layout = QGridLayout(self.processing_group_box)
        processing_layout.setContentsMargins(12, 12, 12, 12)
        processing_layout.setHorizontalSpacing(10)
        processing_layout.setVerticalSpacing(8)
        denoise_label = QLabel("去噪算法", self.processing_group_box)
        threshold_label = QLabel("阈值 (μs)", self.processing_group_box)
        processing_layout.addWidget(denoise_label, 0, 0)
        processing_layout.addWidget(self.roi_settings_editor.noise_filter_combo_box, 0, 1)
        processing_layout.addWidget(threshold_label, 1, 0)
        processing_layout.addWidget(self.roi_settings_editor.noise_threshold_spin_box, 1, 1)
        processing_layout.setColumnStretch(1, 1)

        self.display_group_box = QGroupBox(self.control_panel_widget)
        display_layout = QGridLayout(self.display_group_box)
        display_layout.setContentsMargins(12, 12, 12, 12)
        display_layout.addWidget(self.palette_text_label, 0, 0)
        display_layout.addWidget(self.palette_combo_box, 0, 1)
        display_layout.setColumnStretch(1, 1)

        self.roi_group_box = QGroupBox(self.control_panel_widget)
        roi_layout = QGridLayout(self.roi_group_box)
        roi_layout.setContentsMargins(12, 12, 12, 12)
        roi_layout.setHorizontalSpacing(10)
        roi_layout.setVerticalSpacing(8)
        roi_fields = (
            ("X", self.roi_settings_editor.X_edit),
            ("Y", self.roi_settings_editor.Y_edit),
            ("宽度", self.roi_settings_editor.Width_edit),
            ("高度", self.roi_settings_editor.Height_edit),
        )
        for row, (text, editor) in enumerate(roi_fields, start=1):
            editor.setMinimumWidth(0)
            roi_layout.addWidget(QLabel(text, self.roi_group_box), row, 0)
            roi_layout.addWidget(editor, row, 1)
        roi_layout.addWidget(self.roi_settings_editor.select_roi_button, 5, 0, 1, 2)
        roi_layout.setColumnStretch(1, 1)

        self.playback_display_roi_group_box = QGroupBox(self.control_panel_widget)
        view_controls_layout = QVBoxLayout(self.playback_display_roi_group_box)
        view_controls_layout.setContentsMargins(10, 8, 10, 10)
        view_controls_layout.setSpacing(8)
        for group_box, title in (
            (self.playback_group_box, "采集与回放"),
            (self.display_group_box, "显示设置"),
            (self.roi_group_box, "ROI 区域"),
        ):
            group_box.setTitle(title)
            group_box.setProperty("uiRole", "controlSubsection")
            view_controls_layout.addWidget(group_box)

    def _init_control_panel_accordion(self):
        """Turn the existing control groups into a compact accordion."""
        sections = (
            (self.source_group_box, "数据源"),
            (self.playback_display_roi_group_box, "采集、显示与 ROI"),
            (self.recording_group_box, "数据录制"),
            (self.model_prediction_group_box, "模型与预测"),
            (self.processing_group_box, "去噪"),
        )
        self._control_accordion_sections = []

        for group_box, title in sections:
            self.control_panel_layout.removeWidget(group_box)
            group_box.setTitle("")
            group_box.setProperty("uiRole", "accordionContent")

            section = QFrame(self.control_panel_widget)
            section.setProperty("uiRole", "accordionSection")
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(0)

            header = QPushButton(title, section)
            header.setProperty("uiRole", "accordionHeader")
            header.setCheckable(True)
            header.setCursor(Qt.CursorShape.PointingHandCursor)
            header.setMinimumHeight(54)
            header.clicked.connect(
                lambda checked, target=group_box: self._set_accordion_section(target, checked)
            )

            section_layout.addWidget(header)
            section_layout.addWidget(group_box)
            self._control_accordion_sections.append((header, group_box))

        spacer_index = self.control_panel_layout.count() - 1
        for offset, (section, _) in enumerate(
            (entry[0].parentWidget(), entry[1]) for entry in self._control_accordion_sections
        ):
            self.control_panel_layout.insertWidget(spacer_index + offset, section)

        self.control_panel_layout.setSpacing(0)
        self._set_accordion_section(self.source_group_box, True)

    def _set_accordion_section(self, target, expanded):
        for header, content in self._control_accordion_sections:
            is_target = content is target
            is_expanded = bool(expanded) if is_target else False
            header.blockSignals(True)
            header.setChecked(is_expanded)
            header.blockSignals(False)
            content.setVisible(is_expanded)
        self.control_panel_layout.activate()
        self.control_panel_widget.setMinimumHeight(
            self.control_panel_layout.sizeHint().height()
        )
        self.control_panel_widget.updateGeometry()

    def _apply_source_mode(self):
        """Apply all live-versus-file visibility from one source of truth."""
        file_mode = source_is_file(self.controller)
        input_path = str(self.controller.input_file_path or "").lower()
        if not file_mode:
            control_title = "实时采集"
        elif input_path.endswith(".raw"):
            control_title = "RAW 回放"
        else:
            control_title = "文件回放"
        self.playback_group_box.setTitle(control_title)

        for widget in (
            self.speed_text_label,
            self.replay_speed_combo_box,
            self.playback_progress_slider,
            self.playback_time_label,
        ):
            widget.setVisible(file_mode)

        recording_header = None
        for header, content in self._control_accordion_sections:
            if content is self.recording_group_box:
                recording_header = header
                header.parentWidget().setVisible(not file_mode)
                break
        self.record_button.setVisible(not file_mode)
        if file_mode and recording_header is not None and recording_header.isChecked():
            self._set_accordion_section(self.source_group_box, True)

        for button, selected in (
            (self.live_camera_button, not file_mode),
            (self.select_input_file_button, file_mode),
        ):
            button.setProperty("sourceSelected", selected)
            button.style().unpolish(button)
            button.style().polish(button)

        if not file_mode:
            self._reset_playback_progress()
        self.control_panel_layout.activate()
        self.control_panel_widget.setMinimumHeight(
            self.control_panel_layout.sizeHint().height()
        )
        self.control_panel_widget.updateGeometry()
        QTimer.singleShot(0, self._fit_event_view)

    def _init_log_panel_ui(self):
        self._log_collapsed = False
        self.log_group_box.setTitle("")
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMinimumHeight(72)
        self.log_text_edit.setMaximumHeight(110)

        log_header = QWidget(self.log_group_box)
        log_header.setObjectName("log_header_widget")
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(2, 0, 2, 0)
        log_header_layout.setSpacing(6)

        log_title = QLabel("运行日志", log_header)
        log_title.setObjectName("log_title_label")
        log_header_layout.addWidget(log_title)
        log_header_layout.addStretch(1)

        self.clear_log_button = QPushButton("清空", log_header)
        self.clear_log_button.setObjectName("clear_log_button")
        self.log_toggle_button = QPushButton("收起", log_header)
        self.log_toggle_button.setObjectName("log_toggle_button")
        log_header_layout.addWidget(self.clear_log_button)
        log_header_layout.addWidget(self.log_toggle_button)
        self.log_group_layout.insertWidget(0, log_header)

    @staticmethod
    def _create_status_chip(text, state, parent):
        label = QLabel(text, parent)
        label.setProperty("uiRole", "statusChip")
        label.setProperty("statusState", state)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    @staticmethod
    def _create_settings_panel_icon():
        """Draw a small sliders icon without relying on an external asset."""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(QColor("#ffffff"))
        for y, knob_x in ((6, 15), (12, 9), (18, 14)):
            painter.drawLine(3, y, 21, y)
            painter.drawEllipse(knob_x - 2, y - 2, 4, 4)
        painter.end()
        return QIcon(pixmap)

    def _set_control_panel_visible(self, visible):
        visible = bool(visible)
        self.control_panel_scroll_area.setVisible(visible)

        if self.settings_panel_button.isChecked() != visible:
            self.settings_panel_button.blockSignals(True)
            self.settings_panel_button.setChecked(visible)
            self.settings_panel_button.blockSignals(False)
        self.settings_panel_button.setToolTip(
            "收起右侧设置面板" if visible else "展开右侧设置面板"
        )

        self.content_horizontal_layout.activate()
        QTimer.singleShot(0, self._fit_event_view)
        QTimer.singleShot(0, self._elide_input_file_name)

    def _connect_signals(self):
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.roi_settings_editor.settings_confirmed.connect(self.on_settings_confirmed)
        self.roi_settings_editor.noise_filter_combo_box.currentTextChanged.connect(
            self._update_noise_threshold_enabled
        )
        self.palette_combo_box.currentTextChanged.connect(self.update_display_settings)
        self.fps_spin_box.valueChanged.connect(self.update_display_settings)
        self.replay_speed_combo_box.currentTextChanged.connect(self.update_replay_speed)
        self.select_weight_button.clicked.connect(self.select_weight_file)
        self.load_model_button.clicked.connect(self.load_eventmamba)
        self.unload_model_button.clicked.connect(self.unload_eventmamba)
        self.restart_model_button.clicked.connect(self.restart_eventmamba)
        self.live_camera_button.clicked.connect(self.select_live_camera)
        self.select_input_file_button.clicked.connect(self.select_input_file)
        self.playback_progress_slider.sliderPressed.connect(self._begin_progress_drag)
        self.playback_progress_slider.sliderMoved.connect(self._preview_progress_drag)
        self.playback_progress_slider.sliderReleased.connect(self._finish_progress_drag)
        self.clear_log_button.clicked.connect(self.log_text_edit.clear)
        self.log_toggle_button.clicked.connect(self._toggle_log_panel)
        self.settings_panel_button.toggled.connect(self._set_control_panel_visible)
        self.controller.connect_view(
            self._display_image_with_prediction,
            self.append_log,
            self._buffer_prediction_result,
            self.handle_playback_finished,
            self.handle_playback_progress,
        )

    def _init_view_state(self):
        self.log_text_edit.document().setMaximumBlockCount(500)
        self.replay_speed_combo_box.setCurrentText("1x")
        self.weight_path_label.setToolTip(self.weight_path_label.text())
        self.input_file_label.setToolTip(self.input_file_label.text())
        self.view_state.set_live_camera()
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.view_state.set_model_stopped()
        self._set_status_chip(
            self.mode_status_label,
            mode_display_name(self.settings.prediction_mode),
            "info",
        )
        self._reset_playback_progress()
        self._update_noise_threshold_enabled(
            self.roi_settings_editor.noise_filter_combo_box.currentText()
        )

    def _update_noise_threshold_enabled(self, filter_name):
        enabled = str(filter_name).strip().lower() != "none"
        self.roi_settings_editor.noise_threshold_spin_box.setEnabled(enabled)

    def append_log(self, message, level=None):
        message = str(message)
        level = level or log_level_for_message(message)
        colors = {
            "default": "#dbeafe",
            "info": "#93c5fd",
            "success": "#86efac",
            "warning": "#fbbf24",
            "error": "#fca5a5",
        }
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(colors.get(level, colors["default"])))

        cursor = self.log_text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"{message}\n", text_format)
        self.log_text_edit.setTextCursor(cursor)
        self.log_text_edit.ensureCursorVisible()

    def set_runtime_status(self, target, text, state="idle"):
        label = getattr(self, f"{target}_status_label", None)
        if label is not None:
            self._set_status_chip(label, text, state)

    def set_source_status(self, file_path):
        self._last_frame_size = None
        self._apply_source_mode()
        if not source_is_file(self.controller):
            self._input_file_display_name = "实时相机"
            self.input_file_label.setText("实时相机")
            self.input_file_label.setToolTip("使用已连接的实时事件相机")
            self._set_status_chip(self.source_status_label, "实时输入", "info")
            return

        if not self.controller.is_camera_running():
            self.start_camera_button.setText("开始播放")
        normalized_path = str(file_path or "").replace("\\", "/")
        full_name = normalized_path.rsplit("/", 1)[-1] or self.input_file_label.text()
        self._input_file_display_name = full_name
        self.input_file_label.setToolTip(
            f"完整文件名：{full_name}\n完整路径：{file_path}"
        )
        QTimer.singleShot(0, self._elide_input_file_name)
        self._set_status_chip(
            self.source_status_label,
            source_display_name(file_path),
            "info",
        )

    @staticmethod
    def _set_status_chip(label, text, state):
        label.setText(str(text))
        label.setProperty("statusState", state)
        label.style().unpolish(label)
        label.style().polish(label)

    def _toggle_log_panel(self):
        self._log_collapsed = not self._log_collapsed
        self.log_text_edit.setVisible(not self._log_collapsed)
        self.log_toggle_button.setText("展开" if self._log_collapsed else "收起")
        if self._log_collapsed:
            # Let Qt account for the active font, DPI scaling, group-box
            # stylesheet margin and layout margins. A fixed 48 px height leaves
            # too little room for the header on some Windows display scales and
            # clips the lower part of Chinese glyphs.
            self.log_group_layout.activate()
            collapsed_height = self.log_group_box.minimumSizeHint().height()
            self.log_group_box.setFixedHeight(collapsed_height)
        else:
            self.log_group_box.setMinimumHeight(0)
            self.log_group_box.setMaximumHeight(16777215)
        self.log_group_box.updateGeometry()
        QTimer.singleShot(0, self._fit_event_view)

    def toggle_camera(self):
        if not self.controller.is_camera_running():
            self._sync_capture_settings_from_ui()
            self.controller.start_camera()
            self.view_state.set_camera_running()
        else:
            self.stop_camera()

    def toggle_recording(self):
        recording_started = self.controller.toggle_recording()
        if recording_started is None:
            return
        if recording_started:
            self.view_state.set_recording_running()
        else:
            self.view_state.set_recording_stopped(enabled=True)

    def _display_image_with_prediction(self, cv_img, img_timestamp):
        if hasattr(cv_img, "flags") and not cv_img.flags["C_CONTIGUOUS"]:
            cv_img = cv_img.copy()

        if len(cv_img.shape) == 3:
            height, width, channel = cv_img.shape
            bytes_per_line = channel * width
            img_format = QImage.Format.Format_BGR888
        else:
            height, width = cv_img.shape
            bytes_per_line = width
            img_format = QImage.Format.Format_Grayscale8

        frame_size = (width, height)
        if frame_size != self._last_frame_size:
            self._last_frame_size = frame_size
            self._fit_event_view()
            source_path = (
                self.controller.input_file_path
                if source_is_file(self.controller)
                else None
            )
            source_name = source_display_name(source_path)
            self._set_status_chip(
                self.source_status_label,
                f"{source_name}  {width}x{height}",
                "info",
            )

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)
        matched_prediction = self.predictions.match_frame(img_timestamp)
        if matched_prediction is not None:
            draw_prediction(q_img, matched_prediction, width, height, self.settings.roi)

        pixmap = QPixmap.fromImage(q_img)
        self.camera_image_label.setPixmap(
            pixmap.scaled(
                self.camera_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "camera_viewport_widget"):
            QTimer.singleShot(0, self._fit_event_view)
        if hasattr(self, "input_file_label"):
            QTimer.singleShot(0, self._elide_input_file_name)

    def showEvent(self, event):
        super().showEvent(event)
        # Refit after the first real layout pass. Before show(), Qt reports
        # small placeholder sizes for widgets whose sidebar is hidden.
        QTimer.singleShot(0, self._fit_event_view)

    def _elide_input_file_name(self):
        if not hasattr(self, "_input_file_display_name"):
            return
        available_width = max(40, self.input_file_label.contentsRect().width() - 16)
        display_text = self.input_file_label.fontMetrics().elidedText(
            self._input_file_display_name,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        self.input_file_label.setText(display_text)

    def _fit_event_view(self):
        if not hasattr(self, "camera_viewport_widget"):
            return
        available = self.camera_viewport_widget.contentsRect().size()
        if available.width() <= 0 or available.height() <= 0:
            return
        # During the first layout pass Qt can briefly report the viewport at
        # only a few pixels. Do not lock the image and aligned controls to that
        # transient size; wait until the real window geometry is available.
        if available.width() < 320 or available.height() < 180:
            return

        # The black canvas fills the workspace. Actual frames are still scaled
        # with KeepAspectRatio in _display_image_with_prediction(), so a 16:9
        # source is never stretched even when the surrounding UI is wider.
        target_width = max(1, available.width())
        target_height = max(1, available.height())

        if self.camera_image_label.size() != QSize(target_width, target_height):
            self.camera_image_label.setFixedSize(target_width, target_height)
        for aligned_widget in (
            self.viewer_header_widget,
            self.playback_progress_widget,
        ):
            if aligned_widget.width() != target_width:
                aligned_widget.setFixedWidth(target_width)

    def _buffer_prediction_result(self, result, pred_timestamp):
        self.append_log(backend_message(result))
        self.predictions.add_result(result, pred_timestamp, self.settings.prediction_mode)

    def closeEvent(self, event):
        if self._close_ready:
            event.accept()
            return

        event.ignore()
        if not self._close_pending:
            self._close_pending = True
            self.setEnabled(False)
            self._inference_health_timer.stop()

        if self._inference_operation_is_running():
            if self._inference_operation.operation_name == INFERENCE_CLOSE:
                # Do not cancel the cleanup worker itself. If it were
                # interrupted before calling stop_backend(), accepting this
                # close would leave the inference process orphaned.
                return
            self._inference_operation.requestInterruption()
            try:
                self.controller.cancel_model_start()
            except Exception as exc:
                self.append_log(f"取消推理启动失败：{exc}", "error")
            return

        self._begin_close_cleanup()

    def _begin_close_cleanup(self):
        if self._inference_operation_is_running():
            return

        try:
            self.controller.close_ui_resources()
        except Exception as exc:
            self.append_log(f"关闭 Qt 运行资源失败：{exc}", "error")
            self._close_pending = False
            self.setEnabled(True)
            self._inference_health_timer.start()
            self.view_state.set_model_error()
            return

        self.view_state.set_model_stopping()
        started = self._start_inference_operation(
            INFERENCE_CLOSE,
            self.controller.close_backend_resources,
            allow_when_closing=True,
        )
        if not started:
            # Starting the cleanup worker can itself fail. Keep the window
            # alive so the retained process/thread handles can be retried.
            if self._close_pending:
                self.append_log("无法启动关闭清理任务，窗口保持打开", "error")
                self._close_pending = False
                self.setEnabled(True)
                self._inference_health_timer.start()
                self.view_state.set_model_error()

    def _complete_close(self):
        self._close_ready = True
        QTimer.singleShot(0, self.close)

    def update_replay_speed(self):
        self._sync_capture_settings_from_ui()
        self.controller.update_replay_factor()

    def update_display_settings(self):
        self._sync_capture_settings_from_ui()
        self.controller.update_display_settings()

    def handle_playback_finished(self):
        self.stop_camera()

    def handle_playback_progress(self, current_us, total_us):
        if not source_is_file(self.controller):
            self._reset_playback_progress()
            return

        total_us = max(0, int(total_us or 0))
        current_us = max(0, int(current_us or 0))
        if total_us > 0:
            current_us = min(total_us, current_us)
        self._progress_total_us = total_us
        self.playback_progress_slider.setEnabled(total_us > 0)

        if not self._is_dragging_progress:
            value = 0
            if total_us > 0:
                value = int((current_us / total_us) * PLAYBACK_SLIDER_MAX)
            self.playback_progress_slider.blockSignals(True)
            self.playback_progress_slider.setValue(value)
            self.playback_progress_slider.blockSignals(False)

        if total_us > 0:
            total_label = _format_playback_time_us(total_us)
        else:
            total_label = "--:--"
        self.playback_time_label.setText(f"{_format_playback_time_us(current_us)} / {total_label}")

    def stop_camera(self):
        self.controller.stop_camera()
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)
        self.camera_image_label.setText(
            "回放已停止" if source_is_file(self.controller) else "相机未启动"
        )
        self.predictions.clear()
        self._reset_playback_progress()

    def select_live_camera(self):
        if not source_is_file(self.controller):
            self.view_state.set_live_camera()
            self._refresh_camera_view_state()
            return

        self._reset_playback_progress()
        self.predictions.clear()
        QApplication.processEvents()
        self._sync_capture_settings_from_ui()
        self.controller.set_live_camera()
        self.view_state.set_live_camera()
        self._refresh_camera_view_state()

    def select_input_file(self):
        file_path = choose_input_file(self)
        if not file_path:
            return

        self._reset_playback_progress()
        self.predictions.clear()
        QApplication.processEvents()
        self._sync_capture_settings_from_ui()
        self.controller.set_input_file(file_path, restart_if_running=True)
        self.view_state.set_input_file(file_path)
        self._refresh_camera_view_state()

    def _refresh_camera_view_state(self):
        if self.controller.is_camera_running():
            self.view_state.set_camera_running()
            return
        self.view_state.set_camera_stopped()
        self.view_state.set_recording_stopped(enabled=False)

    def select_weight_file(self):
        if self.controller.is_inference_running():
            self.append_log("请先停止推理服务，再选择其他模型", "warning")
            return
        weights_path = choose_weights_file(
            self,
            runtime_kind=self.controller.inference_runtime_kind,
        )
        if not weights_path:
            return

        self.view_state.set_weight_file(weights_path)
        self.controller.set_weights_path(weights_path)

    def load_eventmamba(self):
        if self._close_pending:
            return
        if self._inference_operation_is_running():
            self.append_log("已有推理服务操作正在进行，请稍候", "warning")
            return
        if self.controller.weights_path is None:
            self.append_log("请先选择模型文件", "warning")
            return
        if not self._stop_model_network_before_backend("启动"):
            return

        runtime_name = self.controller.inference_runtime_display_name
        self.view_state.set_model_starting()
        self.append_log(
            f"正在启动 {runtime_name} 推理服务并加载{mode_display_name(self.settings.prediction_mode)}模型，首次加载可能需要几秒钟...",
            "info",
        )
        self._start_inference_operation(INFERENCE_START, self.controller.load_model)

    def unload_eventmamba(self):
        if self._close_pending:
            return
        if self._inference_operation_is_running():
            self.append_log("已有推理服务操作正在进行，请稍候", "warning")
            return
        runtime_name = self.controller.inference_runtime_display_name
        self.view_state.set_model_stopping()
        if not self._stop_model_network_before_backend("停止"):
            return
        self.append_log(f"正在停止 {runtime_name} 推理服务...", "info")
        self._start_inference_operation(INFERENCE_STOP, self.controller.unload_model)

    def restart_eventmamba(self):
        if self._close_pending:
            return
        if self._inference_operation_is_running():
            self.append_log("已有推理服务操作正在进行，请稍候", "warning")
            return
        if self.controller.weights_path is None:
            self.append_log("请先选择模型文件", "warning")
            return
        runtime_name = self.controller.inference_runtime_display_name
        self.view_state.set_model_starting()
        if not self._stop_model_network_before_backend("重启"):
            return
        self.append_log(f"正在重启 {runtime_name} 推理服务...", "info")
        self._start_inference_operation(INFERENCE_RESTART, self.controller.restart_model)

    def _stop_model_network_before_backend(self, action):
        """Destroy NetworkThread on the UI thread before a backend worker."""
        try:
            self.controller.stop_model_network()
        except Exception as exc:
            self.view_state.set_model_error()
            self.append_log(f"{action}推理前停止网络线程失败：{exc}", "error")
            self._last_inference_state = self.controller.inference_state
            return False
        return True

    def _start_inference_operation(
        self,
        operation_name,
        operation,
        allow_when_closing=False,
    ):
        if self._close_pending and not allow_when_closing:
            return False
        if self._inference_operation_is_running():
            self.append_log("已有推理服务操作正在进行，请稍候", "warning")
            return False

        worker = InferenceOperationThread(operation_name, operation, self)
        worker.succeeded.connect(self._handle_inference_operation_success)
        worker.failed.connect(self._handle_inference_operation_failure)
        worker.cancelled.connect(self._handle_inference_operation_cancelled)
        worker.finished.connect(self._finish_inference_operation)
        self._inference_operation = worker
        self._set_prediction_mode_controls_enabled(False)
        try:
            worker.start()
        except Exception as exc:
            self._inference_operation = None
            worker.deleteLater()
            self._set_prediction_mode_controls_enabled(True)
            self._handle_inference_operation_failure(operation_name, str(exc))
            return False
        return True

    def _handle_inference_operation_success(self, operation_name):
        if self._close_pending or operation_name == INFERENCE_CLOSE:
            return

        runtime_name = self.controller.inference_runtime_display_name
        if operation_name == INFERENCE_STOP:
            self.view_state.set_model_stopped()
            self.predictions.clear()
            self.append_log(f"{runtime_name} 推理服务已停止", "info")
        elif operation_name == INFERENCE_CLEANUP:
            self.view_state.set_model_stopped()
            self.predictions.clear()
            self.append_log("失败操作残留的推理后端已清理", "info")
        elif operation_name in (INFERENCE_START, INFERENCE_RESTART):
            try:
                self.controller.start_model_network()
            except Exception as exc:
                self._handle_network_start_failure(operation_name, exc)
                return
            if self.controller.active_model_path:
                self.view_state.set_weight_file(self.controller.active_model_path)
            self.view_state.set_model_running()
            action = "启动" if operation_name == INFERENCE_START else "重启"
            self.append_log(f"{runtime_name} 推理服务已{action}", "success")
        self._last_inference_state = self.controller.inference_state

    def _handle_network_start_failure(self, operation_name, error):
        cleanup_details = ""
        try:
            self.controller.stop_model_network()
        except Exception as cleanup_error:
            cleanup_details = f"；网络线程清理失败：{cleanup_error}"

        self._cleanup_after_operation = True
        self.view_state.set_model_error()
        action = "启动" if operation_name == INFERENCE_START else "重启"
        self.append_log(
            f"{action}推理网络失败：{error}{cleanup_details}；正在清理后端",
            "error",
        )
        self._last_inference_state = self.controller.inference_state

    def _handle_inference_operation_failure(self, operation_name, message):
        action = {
            INFERENCE_START: "启动推理",
            INFERENCE_STOP: "停止推理",
            INFERENCE_RESTART: "重启推理",
            INFERENCE_CLEANUP: "清理推理后端",
            INFERENCE_CLOSE: "关闭推理后端",
        }.get(operation_name, "推理操作")
        self.append_log(f"{action}失败：{message}", "error")

        if operation_name == INFERENCE_CLOSE:
            self._close_pending = False
            self.setEnabled(True)
            self._inference_health_timer.start()
            self.view_state.set_model_error()
            self._last_inference_state = self.controller.inference_state
            return
        if self._close_pending:
            return
        self.view_state.set_model_error()
        self._last_inference_state = self.controller.inference_state

    def _handle_inference_operation_cancelled(self, operation_name):
        if operation_name == INFERENCE_CLOSE:
            self._close_pending = False
            self.setEnabled(True)
            self._inference_health_timer.start()
            self.view_state.set_model_error()
            self.append_log("关闭清理被取消，窗口保持打开", "warning")
            self._last_inference_state = self.controller.inference_state
            return
        if self._close_pending:
            return
        self.view_state.set_model_error()
        self.append_log(f"推理操作已取消：{operation_name}", "warning")
        self._last_inference_state = self.controller.inference_state

    def _finish_inference_operation(self):
        worker = self._inference_operation
        operation_name = None
        if worker is not None:
            operation_name = worker.operation_name
            worker.deleteLater()
        self._inference_operation = None

        if self._close_pending:
            self._cleanup_after_operation = False
            if operation_name == INFERENCE_CLOSE:
                self._complete_close()
            else:
                QTimer.singleShot(0, self._begin_close_cleanup)
            return

        if self._cleanup_after_operation:
            self._cleanup_after_operation = False
            self.view_state.set_model_stopping()
            QTimer.singleShot(0, self._start_backend_cleanup)
            return

        self._set_prediction_mode_controls_enabled(True)

    def _start_backend_cleanup(self):
        if self._close_pending:
            self._begin_close_cleanup()
            return
        started = self._start_inference_operation(
            INFERENCE_CLEANUP,
            self.controller.unload_model,
        )
        if not started:
            self._set_prediction_mode_controls_enabled(True)
            self.view_state.set_model_error()
            self.append_log("无法启动推理后端清理任务", "error")

    def _inference_operation_is_running(self):
        # Keep the UI operation busy until its queued success/failure and
        # finished signals have all been handled on the main thread. A worker
        # can already report isRunning() == False while those signals are
        # still pending.
        return self._inference_operation is not None

    def _set_prediction_mode_controls_enabled(self, enabled):
        self.roi_settings_editor.center_radioButton.setEnabled(enabled)
        self.roi_settings_editor.eli_radioButton.setEnabled(enabled)

    def _refresh_inference_state(self):
        if self._close_pending or self._inference_operation_is_running():
            return
        if self.controller.inference_state == "running":
            self.controller.is_inference_running()
        state = self.controller.inference_state
        if state == self._last_inference_state:
            return
        self._last_inference_state = state

        if state == "running":
            self.view_state.set_model_running()
        elif state == "starting":
            self.view_state.set_model_starting()
        elif state == "stopping":
            self.view_state.set_model_stopping()
        elif state == "error":
            self.view_state.set_model_error()
            if self.controller.inference_last_error:
                self.append_log(
                    f"推理服务异常：{self.controller.inference_last_error}",
                    "error",
                )
        else:
            self.view_state.set_model_stopped()

    def on_settings_confirmed(self, roi, mode, filter_type, threshold_us):
        if roi is not None and self._last_frame_size is not None:
            frame_width, frame_height = self._last_frame_size
            if normalize_roi(roi, frame_width, frame_height) is None:
                QMessageBox.warning(
                    self,
                    "区域无效",
                    "ROI 与当前图像没有交集，请检查 X、Y、宽度和高度。",
                )
                return
        inference_mode_changed = mode != self.settings.prediction_mode
        previous_roi = self.settings.roi
        self._sync_capture_settings_from_ui()
        camera_settings_changed = self.controller.apply_settings(
            roi,
            mode,
            filter_type,
            threshold_us,
        )
        if camera_settings_changed and self.controller.is_camera_running():
            QApplication.processEvents()

        self.append_log(
            noise_settings_message(filter_type, self.settings.noise_filter_threshold_us),
            "info",
        )
        applied_roi = self.settings.roi
        if applied_roi != previous_roi:
            self.predictions.clear()
        if applied_roi is not None or (roi is None and previous_roi is not None):
            self.append_log(roi_settings_message(applied_roi, mode), "info")
        self._set_status_chip(self.mode_status_label, mode_display_name(mode), "info")
        if inference_mode_changed and self.controller.is_inference_running():
            self.restart_eventmamba()

    def _selected_palette(self):
        selected = self.palette_combo_box.currentText()
        if selected in SUPPORTED_PALETTES:
            return selected
        return "Dark"

    def _sync_capture_settings_from_ui(self):
        self.controller.sync_capture_settings(
            self._selected_palette(),
            self.fps_spin_box.value(),
            self._selected_replay_factor(),
        )

    def _selected_replay_factor(self):
        return REPLAY_SPEEDS.get(self.replay_speed_combo_box.currentText(), 1.0)

    def _begin_progress_drag(self):
        if self._progress_total_us <= 0:
            return
        self._is_dragging_progress = True

    def _preview_progress_drag(self, value):
        if self._progress_total_us <= 0:
            return
        value = max(0, min(PLAYBACK_SLIDER_MAX, int(value)))
        current_us = int((value / PLAYBACK_SLIDER_MAX) * self._progress_total_us)
        self.playback_time_label.setText(
            f"{_format_playback_time_us(current_us)} / {_format_playback_time_us(self._progress_total_us)}"
        )

    def _finish_progress_drag(self):
        if self._progress_total_us <= 0:
            self._is_dragging_progress = False
            return

        target_value = self.playback_progress_slider.sliderPosition()
        target_value = max(0, min(PLAYBACK_SLIDER_MAX, int(target_value)))

        self._is_dragging_progress = False
        self.playback_progress_slider.blockSignals(True)
        self.playback_progress_slider.setValue(target_value)
        self.playback_progress_slider.blockSignals(False)
        seek_fraction = target_value / PLAYBACK_SLIDER_MAX
        self._sync_capture_settings_from_ui()
        self.predictions.clear()
        self.controller.seek_playback(seek_fraction)

    def _reset_playback_progress(self):
        self._progress_total_us = 0
        self._is_dragging_progress = False
        self.playback_progress_slider.blockSignals(True)
        self.playback_progress_slider.setValue(0)
        self.playback_progress_slider.blockSignals(False)
        self.playback_progress_slider.setEnabled(False)
        self.playback_time_label.setText("--:-- / --:--")


def _format_playback_time_us(timestamp_us):
    timestamp_us = max(0, int(timestamp_us or 0))
    seconds = timestamp_us / 1_000_000.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    second_value = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{second_value:06.3f}"
    return f"{minutes:02d}:{second_value:06.3f}"
