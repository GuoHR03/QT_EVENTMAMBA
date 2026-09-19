"""Present event frames and keep viewport geometry out of MainWindow."""

from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap

from .prediction_overlay import draw_prediction
from .ui_status import source_display_name
from .view_state import source_is_file


@dataclass(frozen=True)
class EventViewportPorts:
    controller: Any
    predictions: Any
    settings: Any
    camera_image_label: Any
    source_status_label: Any
    camera_viewport_widget: Any
    viewer_header_widget: Any
    playback_progress_widget: Any
    input_file_label: Any
    set_status_chip: Callable[[Any, str, str], None]
    frame_overlays: tuple = ()


class EventViewportPresenter:
    def __init__(self, ports):
        self.ports = ports
        self.frame_size = None
        self.input_file_display_name = None

    def display_image(self, cv_img, img_timestamp):
        ports = self.ports
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
        if frame_size != self.frame_size:
            self.frame_size = frame_size
            self.fit()
            source_path = (
                ports.controller.input_file_path
                if source_is_file(ports.controller)
                else None
            )
            source_name = source_display_name(source_path)
            ports.set_status_chip(
                ports.source_status_label,
                f"{source_name}  {width}x{height}",
                "info",
            )

        q_img = QImage(cv_img.data, width, height, bytes_per_line, img_format)
        matched_prediction = ports.predictions.match_frame(img_timestamp)
        if matched_prediction is not None:
            draw_prediction(
                q_img,
                matched_prediction,
                width,
                height,
                ports.settings.roi,
            )
        for overlay in ports.frame_overlays:
            overlay.draw(q_img, img_timestamp, width, height)

        pixmap = QPixmap.fromImage(q_img)
        ports.camera_image_label.setPixmap(
            pixmap.scaled(
                ports.camera_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def elide_input_file_name(self):
        ports = self.ports
        if self.input_file_display_name is None:
            return
        available_width = max(
            40,
            ports.input_file_label.contentsRect().width() - 16,
        )
        display_text = ports.input_file_label.fontMetrics().elidedText(
            self.input_file_display_name,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        ports.input_file_label.setText(display_text)

    def reset_frame_size(self):
        self.frame_size = None

    def set_input_file_display_name(self, name):
        self.input_file_display_name = str(name)

    def fit(self):
        ports = self.ports
        available = ports.camera_viewport_widget.contentsRect().size()
        if available.width() <= 0 or available.height() <= 0:
            return
        if available.width() < 320 or available.height() < 180:
            return

        target_width = max(1, available.width())
        target_height = max(1, available.height())
        if ports.camera_image_label.size() != QSize(target_width, target_height):
            ports.camera_image_label.setFixedSize(target_width, target_height)
        for aligned_widget in (
            ports.viewer_header_widget,
            ports.playback_progress_widget,
        ):
            if aligned_widget.width() != target_width:
                aligned_widget.setFixedWidth(target_width)
