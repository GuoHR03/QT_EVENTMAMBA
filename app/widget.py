import time

from PyQt6 import uic
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import (
    QColor,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMessageBox,
    QWidget,
)

from backend.event_processing import normalize_roi
from backend.playback_config import PlaybackConfig

from .bootstrap import app_resource_path
from .camera_ui_coordinator import CameraUiCoordinator, CameraUiPorts, REPLAY_SPEEDS
from .controller import AppController
from .event_viewport_presenter import EventViewportPorts, EventViewportPresenter
from .file_dialogs import choose_input_file, choose_weights_file
from .inference_operation_coordinator import (
    InferenceOperationCoordinator,
    InferenceUiPorts,
)
from .inference_operation_state import (
    INFERENCE_CLOSE,
    InferenceOperationState,
)
from .ini30_ground_truth import Ini30GroundTruthOverlay
from .log_formatter import (
    backend_message,
    mode_display_name,
    noise_settings_message,
    roi_settings_message,
)
from .main_window_layout import MainWindowLayoutMixin
from .paths import default_checkpoint_dir, default_onnx_model_dir, default_record_dir
from .performance_metrics import PerformanceMetrics
from .playback_progress import PlaybackProgressState
from .preferences import PreferenceStore, UiPreferences
from .prediction_state import PredictionState
from .settings import AppSettings
from .shutdown_coordinator import ApplicationShutdownCoordinator
from .theme import apply_app_theme
from .ui_log import PredictionLogThrottle, log_level_for_message
from .ui_status import source_display_name
from .view_state import MainViewState, source_is_file


class MainWindow(MainWindowLayoutMixin, QWidget):
    def __init__(self, preference_store=None):
        super().__init__()
        uic.loadUi(app_resource_path("form.ui"), self)
        self.preference_store = preference_store or PreferenceStore()
        self.saved_preferences = self.preference_store.load()
        self.settings = AppSettings(
            prediction_mode=self.saved_preferences.prediction_mode,
            playback_config=PlaybackConfig(
                palette=self.saved_preferences.palette,
                fps=self.saved_preferences.fps,
                replay_factor=REPLAY_SPEEDS[self.saved_preferences.replay_speed],
                noise_filter_type=self.saved_preferences.noise_filter_type,
                noise_filter_threshold_us=(
                    self.saved_preferences.noise_filter_threshold_us
                ),
            ),
        )
        self._init_workspace_ui()
        apply_app_theme(self)
        if not self.preference_store.restore_window_geometry(self):
            self._set_initial_window_geometry()

        self.controller = AppController(self.settings)
        self.ini30_ground_truth = Ini30GroundTruthOverlay()
        self._weight_file_display_name = None
        self._configure_inference_runtime_ui()
        self.view_state = MainViewState(self)
        self.predictions = PredictionState(interval_ms=20)
        self.prediction_log_throttle = PredictionLogThrottle(interval_s=1.0)
        self.performance_metrics = PerformanceMetrics(window_s=5.0)
        self.playback_progress = PlaybackProgressState()
        self.inference_operations = InferenceOperationState()
        self.viewport_presenter = EventViewportPresenter(
            EventViewportPorts(
                controller=self.controller,
                predictions=self.predictions,
                settings=self.settings,
                camera_image_label=self.camera_image_label,
                source_status_label=self.source_status_label,
                camera_viewport_widget=self.camera_viewport_widget,
                viewer_header_widget=self.viewer_header_widget,
                playback_progress_widget=self.playback_progress_widget,
                input_file_label=self.input_file_label,
                set_status_chip=self._set_status_chip,
                frame_overlays=(self.ini30_ground_truth,),
            )
        )
        self.camera_ui = CameraUiCoordinator(
            CameraUiPorts(
                controller=self.controller,
                view_state=self.view_state,
                playback_progress=self.playback_progress,
                predictions=self.predictions,
                palette_combo_box=self.palette_combo_box,
                fps_spin_box=self.fps_spin_box,
                replay_speed_combo_box=self.replay_speed_combo_box,
                camera_image_label=self.camera_image_label,
                playback_progress_slider=self.playback_progress_slider,
                playback_time_label=self.playback_time_label,
                choose_input_file=self._choose_input_file,
                process_events=QApplication.processEvents,
            )
        )
        self.shutdown = ApplicationShutdownCoordinator(
            self.controller,
            self.inference_operations,
        )

        self._connect_signals()
        self._init_view_state()
        self._inference_health_timer = QTimer(self)
        self._inference_health_timer.setInterval(1000)
        self.inference_operation_coordinator = InferenceOperationCoordinator(
            InferenceUiPorts(
                parent=self,
                controller=self.controller,
                view_state=self.view_state,
                predictions=self.predictions,
                settings=self.settings,
                append_log=self.append_log,
                set_prediction_mode_controls_enabled=(
                    self._set_prediction_mode_controls_enabled
                ),
                set_window_enabled=self.setEnabled,
                start_health_timer=self._inference_health_timer.start,
                begin_close_cleanup=self._begin_close_cleanup,
                complete_close=self._complete_close,
                choose_weights_file=self._choose_weights_file,
                shutdown=self.shutdown,
            ),
            state=self.inference_operations,
        )
        self._inference_health_timer.timeout.connect(self._refresh_inference_state)
        self._inference_health_timer.start()
        self._performance_timer = QTimer(self)
        self._performance_timer.setInterval(5000)
        self._performance_timer.timeout.connect(self._report_performance)
        self._performance_timer.start()

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

    def _connect_signals(self):
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.roi_settings_editor.settings_confirmed.connect(self.on_settings_confirmed)
        self.roi_settings_editor.center_radioButton.toggled.connect(
            lambda checked: checked and self._on_prediction_mode_selected("center")
        )
        self.roi_settings_editor.eli_radioButton.toggled.connect(
            lambda checked: checked and self._on_prediction_mode_selected("ellipse")
        )
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
        self.ground_truth_button.toggled.connect(self._toggle_ground_truth)
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
        for control in (
            self.palette_combo_box,
            self.fps_spin_box,
            self.replay_speed_combo_box,
        ):
            control.blockSignals(True)
        try:
            self.palette_combo_box.setCurrentText(self.saved_preferences.palette)
            self.fps_spin_box.setValue(self.saved_preferences.fps)
            self.replay_speed_combo_box.setCurrentText(
                self.saved_preferences.replay_speed
            )
        finally:
            for control in (
                self.palette_combo_box,
                self.fps_spin_box,
                self.replay_speed_combo_box,
            ):
                control.blockSignals(False)
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
        self._set_log_panel_collapsed(self.saved_preferences.log_collapsed)
        self._set_control_panel_visible(
            self.saved_preferences.settings_panel_visible
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
        self.viewport_presenter.reset_frame_size()
        ground_truth_available = self.ini30_ground_truth.configure_source(file_path)
        self.ground_truth_button.blockSignals(True)
        self.ground_truth_button.setChecked(False)
        self.ground_truth_button.blockSignals(False)
        self.ground_truth_button.setEnabled(ground_truth_available)
        self.ground_truth_button.setText("绘制 Ground Truth 椭圆")
        self._apply_source_mode()
        if not source_is_file(self.controller):
            self.viewport_presenter.set_input_file_display_name("实时相机")
            self.input_file_label.setText("实时相机")
            self.input_file_label.setToolTip("使用已连接的实时事件相机")
            self._set_status_chip(self.source_status_label, "实时输入", "info")
            return

        if not self.controller.is_camera_running():
            self.start_camera_button.setText("开始播放")
        normalized_path = str(file_path or "").replace("\\", "/")
        full_name = normalized_path.rsplit("/", 1)[-1] or self.input_file_label.text()
        self.viewport_presenter.set_input_file_display_name(full_name)
        self.input_file_label.setToolTip(
            f"完整文件名：{full_name}\n完整路径：{file_path}"
        )
        QTimer.singleShot(0, self._elide_input_file_name)
        self._set_status_chip(
            self.source_status_label,
            source_display_name(file_path),
            "info",
        )

    def _toggle_ground_truth(self, enabled):
        active = self.ini30_ground_truth.set_enabled(enabled)
        if bool(enabled) != active:
            self.ground_truth_button.blockSignals(True)
            self.ground_truth_button.setChecked(active)
            self.ground_truth_button.blockSignals(False)
        self.ground_truth_button.setText(
            "Ground Truth 椭圆：开" if active else "绘制 Ground Truth 椭圆"
        )

    @staticmethod
    def _set_status_chip(label, text, state):
        label.setText(str(text))
        label.setProperty("statusState", state)
        label.style().unpolish(label)
        label.style().polish(label)

    def _toggle_log_panel(self):
        self._set_log_panel_collapsed(not self._log_collapsed)

    def _set_log_panel_collapsed(self, collapsed):
        self._log_collapsed = bool(collapsed)
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
        return self.camera_ui.toggle_camera()

    def toggle_recording(self):
        return self.camera_ui.toggle_recording()

    def _display_image_with_prediction(self, cv_img, img_timestamp):
        started = time.perf_counter()
        try:
            return self.viewport_presenter.display_image(cv_img, img_timestamp)
        finally:
            self.performance_metrics.record_frame(time.perf_counter() - started)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "camera_viewport_widget"):
            QTimer.singleShot(0, self._fit_event_view)
        if hasattr(self, "control_panel_scroll_area"):
            QTimer.singleShot(0, self._sync_control_panel_content_width)
        if hasattr(self, "input_file_label"):
            QTimer.singleShot(0, self._elide_input_file_name)
        if hasattr(self, "weight_path_label"):
            QTimer.singleShot(0, self._elide_weight_file_name)

    def showEvent(self, event):
        super().showEvent(event)
        # Refit after the first real layout pass. Before show(), Qt reports
        # small placeholder sizes for widgets whose sidebar is hidden.
        QTimer.singleShot(0, self._fit_event_view)

    def _elide_input_file_name(self):
        return self.viewport_presenter.elide_input_file_name()

    def set_weight_file_display_path(self, file_path):
        normalized_path = str(file_path or "").replace("\\", "/")
        self._weight_file_display_name = normalized_path.rsplit("/", 1)[-1]
        self.weight_path_label.setToolTip(
            f"完整文件名：{self._weight_file_display_name}\n完整路径：{file_path}"
        )
        QTimer.singleShot(0, self._elide_weight_file_name)

    def _elide_weight_file_name(self):
        if not self._weight_file_display_name:
            return
        available_width = max(
            40,
            self.weight_path_label.contentsRect().width() - 16,
        )
        display_text = self.weight_path_label.fontMetrics().elidedText(
            self._weight_file_display_name,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        self.weight_path_label.setText(display_text)

    def _fit_event_view(self):
        return self.viewport_presenter.fit()

    def _buffer_prediction_result(self, result, pred_timestamp):
        self.performance_metrics.record_prediction()
        if self.prediction_log_throttle.should_log(result):
            self.append_log(backend_message(result))
        self.predictions.add_result(result, pred_timestamp, self.settings.prediction_mode)

    def _report_performance(self):
        snapshot = self.performance_metrics.snapshot()
        if snapshot.has_activity:
            self.append_log(snapshot.format_log(), "info")

    def closeEvent(self, event):
        if self.shutdown.ready:
            self._save_preferences()
            event.accept()
            return

        event.ignore()
        if self.shutdown.begin():
            self.setEnabled(False)
            self._inference_health_timer.stop()
            self._performance_timer.stop()

        if self._inference_operation_is_running():
            worker = self.inference_operations.worker
            if self.inference_operations.operation_name == INFERENCE_CLOSE:
                # Do not cancel the cleanup worker itself. If it were
                # interrupted before calling stop_backend(), accepting this
                # close would leave the inference process orphaned.
                return
            worker.requestInterruption()
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
            self.shutdown.close_ui_resources()
        except Exception as exc:
            self.append_log(f"关闭 Qt 运行资源失败：{exc}", "error")
            self.shutdown.abort()
            self.setEnabled(True)
            self._inference_health_timer.start()
            self._performance_timer.start()
            self.view_state.set_model_error()
            return

        self.view_state.set_model_stopping()
        started = self._start_inference_operation(
            INFERENCE_CLOSE,
            self.shutdown.close_backend_resources,
            allow_when_closing=True,
        )
        if not started:
            # Starting the cleanup worker can itself fail. Keep the window
            # alive so the retained process/thread handles can be retried.
            if self.shutdown.pending:
                self.append_log("无法启动关闭清理任务，窗口保持打开", "error")
                self.shutdown.abort("无法启动关闭清理任务")
                self.setEnabled(True)
                self._inference_health_timer.start()
                self._performance_timer.start()
                self.view_state.set_model_error()

    def _complete_close(self):
        self.shutdown.complete()
        QTimer.singleShot(0, self.close)

    def update_replay_speed(self):
        return self.camera_ui.update_replay_speed()

    def update_display_settings(self):
        return self.camera_ui.update_display_settings()

    def handle_playback_finished(self):
        return self.camera_ui.handle_playback_finished()

    def handle_playback_progress(self, current_us, total_us):
        return self.camera_ui.handle_playback_progress(current_us, total_us)

    def stop_camera(self):
        return self.camera_ui.stop_camera()

    def select_live_camera(self):
        return self.camera_ui.select_live_camera()

    def select_input_file(self):
        return self.camera_ui.select_input_file()

    def _choose_input_file(self):
        file_path = choose_input_file(
            self,
            initial_dir=self.preference_store.last_input_directory(
                default_record_dir()
            ),
        )
        self.preference_store.remember_input_file(file_path)
        return file_path

    def _choose_weights_file(self):
        runtime_kind = self.controller.inference_runtime_kind
        fallback = (
            default_onnx_model_dir()
            if runtime_kind == "windows"
            else default_checkpoint_dir()
        )
        file_path = choose_weights_file(
            self,
            runtime_kind=runtime_kind,
            initial_dir=self.preference_store.last_model_directory(fallback),
        )
        self.preference_store.remember_model_file(file_path)
        return file_path

    def _save_preferences(self):
        self.camera_ui.sync_capture_settings()
        preferences = UiPreferences(
            palette=self.camera_ui.selected_palette(),
            fps=self.fps_spin_box.value(),
            replay_speed=self.replay_speed_combo_box.currentText(),
            prediction_mode=self.settings.prediction_mode,
            noise_filter_type=self.settings.noise_filter_type,
            noise_filter_threshold_us=self.settings.noise_filter_threshold_us,
            log_collapsed=self._log_collapsed,
            settings_panel_visible=self.control_panel_scroll_area.isVisible(),
        )
        self.preference_store.save(preferences)
        self.preference_store.save_window_geometry(self)

    def _refresh_camera_view_state(self):
        return self.camera_ui.refresh_camera_view_state()

    def select_weight_file(self):
        return self.inference_operation_coordinator.select_weight_file()

    def load_eventmamba(self):
        return self.inference_operation_coordinator.load_model()

    def unload_eventmamba(self):
        return self.inference_operation_coordinator.unload_model()

    def restart_eventmamba(self):
        return self.inference_operation_coordinator.restart_model()

    def _stop_model_network_before_backend(self, action):
        return self.inference_operation_coordinator.stop_network_before_backend(
            action
        )

    def _start_inference_operation(
        self,
        operation_name,
        operation,
        allow_when_closing=False,
    ):
        return self.inference_operation_coordinator.start(
            operation_name,
            operation,
            allow_when_closing=allow_when_closing,
        )

    def _inference_operation_is_running(self):
        # Keep the UI operation busy until its queued success/failure and
        # finished signals have all been handled on the main thread. A worker
        # can already report isRunning() == False while those signals are
        # still pending.
        return self.inference_operation_coordinator.busy

    def _set_prediction_mode_controls_enabled(self, enabled):
        self.roi_settings_editor.center_radioButton.setEnabled(enabled)
        self.roi_settings_editor.eli_radioButton.setEnabled(enabled)

    def _refresh_inference_state(self):
        return self.inference_operation_coordinator.refresh_runtime_state()

    def _on_prediction_mode_selected(self, mode):
        if not self.controller.apply_prediction_mode(mode):
            return
        self.predictions.clear()
        self._set_status_chip(self.mode_status_label, mode_display_name(mode), "info")
        self.append_log(f"预测模式已切换为：{mode_display_name(mode)}", "info")
        if self.controller.is_inference_running():
            self.restart_eventmamba()

    def on_settings_confirmed(self, roi, mode, filter_type, threshold_us):
        frame_size = self.viewport_presenter.frame_size
        if roi is not None and frame_size is not None:
            frame_width, frame_height = frame_size
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
        return self.camera_ui.selected_palette()

    def _sync_capture_settings_from_ui(self):
        return self.camera_ui.sync_capture_settings()

    def _selected_replay_factor(self):
        return self.camera_ui.selected_replay_factor()

    def _begin_progress_drag(self):
        return self.camera_ui.begin_progress_drag()

    def _preview_progress_drag(self, value):
        return self.camera_ui.preview_progress_drag(value)

    def _finish_progress_drag(self):
        return self.camera_ui.finish_progress_drag()

    def _reset_playback_progress(self):
        return self.camera_ui.reset_playback_progress()

    def _apply_playback_progress_view(self, view):
        return self.camera_ui.apply_playback_progress_view(view)
