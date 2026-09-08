"""Dynamic MainWindow layout construction isolated from runtime behavior."""

from PyQt6.QtCore import QSize, QTimer, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .choose_windows import ChooseWindow
from .playback_progress import PLAYBACK_SLIDER_MAX
from .view_state import source_is_file


class MainWindowLayoutMixin:
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
        self.select_input_file_button.setText("选择事件文件")
        self.weight_path_label.setText("尚未选择权重")
        self.input_file_label.setText("实时相机")

        self.palette_combo_box.setToolTip("选择事件极性的显示配色")
        self.replay_speed_combo_box.setToolTip("调整离线文件的回放速度")
        self.live_camera_button.setToolTip("切换到已连接的实时事件相机")
        self.select_input_file_button.setToolTip(
            "选择 RAW、H5/HDF5 或 AEDAT4 事件文件"
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
        model_buttons = QGridLayout()
        model_buttons.setHorizontalSpacing(8)
        model_buttons.setVerticalSpacing(8)
        model_buttons.addWidget(self.load_model_button, 0, 0, 1, 2)
        model_buttons.addWidget(self.unload_model_button, 1, 0)
        model_buttons.addWidget(self.restart_model_button, 1, 1)
        model_buttons.setColumnStretch(0, 1)
        model_buttons.setColumnStretch(1, 1)
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

        self.input_playback_group_box = QGroupBox(self.control_panel_widget)
        input_playback_layout = QVBoxLayout(self.input_playback_group_box)
        input_playback_layout.setContentsMargins(10, 8, 10, 10)
        input_playback_layout.setSpacing(8)
        for group_box, title in (
            (self.source_group_box, "数据源"),
            (self.playback_group_box, "采集与回放"),
        ):
            group_box.setTitle(title)
            group_box.setProperty("uiRole", "controlSubsection")
            input_playback_layout.addWidget(group_box)

        self.display_roi_group_box = QGroupBox(self.control_panel_widget)
        display_roi_layout = QVBoxLayout(self.display_roi_group_box)
        display_roi_layout.setContentsMargins(10, 8, 10, 10)
        display_roi_layout.setSpacing(8)
        for group_box, title in (
            (self.display_group_box, "显示设置"),
            (self.roi_group_box, "ROI 区域"),
        ):
            group_box.setTitle(title)
            group_box.setProperty("uiRole", "controlSubsection")
            display_roi_layout.addWidget(group_box)

    def _init_control_panel_accordion(self):
        """Turn the existing control groups into a compact accordion."""
        sections = (
            (self.input_playback_group_box, "输入与播放"),
            (self.display_roi_group_box, "显示与 ROI"),
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
        self._set_accordion_section(self.input_playback_group_box, True)

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
            self._set_accordion_section(self.input_playback_group_box, True)

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


