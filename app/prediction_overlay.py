import ast
import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen

from backend.event_processing import normalize_roi


LEGACY_RESULT_MARKER = "输出结果为："


@dataclass(frozen=True)
class PredictionSample:
    values: tuple
    coordinate_mode: str
    cropped: bool = False
    prediction_mode: str = "center"
    effective_roi: object = None
    has_effective_roi: bool = False


def parse_prediction_result(result, fallback_mode="center"):
    if isinstance(result, dict):
        return _parse_structured_result(result, fallback_mode)
    if isinstance(result, str):
        return _parse_legacy_result(result, fallback_mode)
    return None


def format_prediction_log(result, mode_display_names):
    if isinstance(result, dict):
        msg_type = result.get("msg_type")
        if msg_type == "PREDICTION":
            values = result.get("values", [])
            cropped_text = "是" if result.get("cropped", False) else "否"
            return f"{LEGACY_RESULT_MARKER}{values}（已裁剪：{cropped_text}）"
        return str(result.get("message", result))

    display_text = str(result)
    for mode, name in mode_display_names.items():
        display_text = display_text.replace(f"预测模式: {mode}", f"预测模式：{name}")
        display_text = display_text.replace(f"预测模式：{mode}", f"预测模式：{name}")
    if "|cropped:" in display_text:
        main_part, cropped_part = display_text.rsplit("|cropped:", 1)
        is_cropped = cropped_part.strip().lower() == "true"
        cropped_text = "是" if is_cropped else "否"
        display_text = f"{main_part.strip()}（已裁剪：{cropped_text}）"
    return display_text


def select_prediction_for_frame(prediction_buffer, frame_timestamp, interval_ms, max_age_us=200000):
    frame_end_time = frame_timestamp
    frame_start_time = frame_end_time - int(interval_ms * 1000)

    matched = None
    for ts_key in sorted(prediction_buffer.keys()):
        if frame_start_time <= ts_key <= frame_end_time:
            matched = prediction_buffer[ts_key]
            break
        if ts_key <= frame_end_time:
            matched = prediction_buffer[ts_key]

    for ts_key in list(prediction_buffer.keys()):
        if ts_key < frame_end_time - max_age_us:
            del prediction_buffer[ts_key]

    return matched


def draw_prediction(q_img, sample, width, height, current_roi):
    roi = sample.effective_roi if sample.has_effective_roi else current_roi
    frame_roi = normalize_roi(roi, width, height)
    pixel = map_prediction_to_pixel(sample, width, height, frame_roi)
    if pixel is None:
        return

    px, py = pixel
    painter = QPainter(q_img)
    pen = QPen(QColor(255, 0, 0))
    pen.setWidth(3)
    painter.setPen(pen)
    painter.setBrush(QColor(255, 0, 0, 80))

    if sample.prediction_mode == "ellipse" and len(sample.values) >= 5:
        _, _, major, minor, angle = sample.values[:5]
        scale_width, scale_height = _ellipse_scale(width, height, sample, frame_roi)
        painter.save()
        painter.translate(px, py)
        painter.rotate(math.degrees(angle))
        painter.drawEllipse(QPointF(0, 0), major * scale_width, minor * scale_height)
        painter.restore()
    else:
        painter.drawEllipse(px - 8, py - 8, 16, 16)

    painter.end()


def map_prediction_to_pixel(sample, width, height, current_roi):
    if sample is None:
        return None
    if sample.has_effective_roi:
        current_roi = sample.effective_roi
    current_roi = normalize_roi(current_roi, width, height)
    if sample.cropped:
        return _map_cropped_prediction_to_pixel(sample.values, width, height, current_roi)
    if sample.coordinate_mode == "norm":
        return _map_normalized_prediction_to_pixel(sample.values, width, height)
    return _map_raw_prediction_to_pixel(sample.values, width, height)


def _parse_structured_result(result, fallback_mode):
    if result.get("msg_type") != "PREDICTION":
        return None
    values = result.get("values")
    return _build_sample(
        values=values,
        cropped=bool(result.get("cropped", False)),
        prediction_mode=result.get("mode") or fallback_mode,
        effective_roi=result.get("effective_roi"),
        has_effective_roi="effective_roi" in result,
    )


def _parse_legacy_result(result, fallback_mode):
    if LEGACY_RESULT_MARKER not in result:
        return None

    parts = result.split(LEGACY_RESULT_MARKER, 1)[1].strip()
    is_cropped = False
    if "|cropped:" in parts:
        main_part, cropped_part = parts.rsplit("|cropped:", 1)
        is_cropped = cropped_part.strip().lower() == "true"
        payload = main_part.strip()
    else:
        payload = parts

    try:
        values = ast.literal_eval(payload)
    except Exception:
        return None

    return _build_sample(values=values, cropped=is_cropped, prediction_mode=fallback_mode)


def _build_sample(
    values,
    cropped,
    prediction_mode,
    effective_roi=None,
    has_effective_roi=False,
):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None

    x, y = values[0], values[1]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None

    if len(values) >= 5 and prediction_mode == "ellipse":
        pred_data = tuple(float(value) for value in values[:5])
    else:
        pred_data = (float(x), float(y))

    coordinate_mode = "norm" if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else "pixel"
    return PredictionSample(
        values=pred_data,
        coordinate_mode=coordinate_mode,
        cropped=bool(cropped),
        prediction_mode=prediction_mode,
        effective_roi=effective_roi,
        has_effective_roi=bool(has_effective_roi),
    )


def _map_cropped_prediction_to_pixel(pred, width, height, current_roi):
    x, y = pred[0], pred[1]
    if current_roi:
        roi_x, roi_y, roi_width, roi_height = current_roi
        px = roi_x + _normalized_offset(x, roi_width)
        py = roi_y + _normalized_offset(y, roi_height)
        return _bounded_pixel(px, py, width, height)

    canonical_x = x * 512 + 96
    canonical_y = y * 512 - 16
    crop_left = math.ceil(96 * width / 640)
    crop_right = math.ceil(608 * width / 640) - 1
    px = max(
        _bounded_coordinate(crop_left, width),
        min(
            _bounded_coordinate(crop_right, width),
            int(canonical_x * width / 640),
        ),
    )
    py = _bounded_coordinate(int(canonical_y * height / 480), height)
    return _bounded_pixel(px, py, width, height)


def _map_normalized_prediction_to_pixel(pred, width, height):
    px = _normalized_offset(pred[0], width)
    py = _normalized_offset(pred[1], height)
    return _bounded_pixel(px, py, width, height)


def _map_raw_prediction_to_pixel(pred, width, height):
    px = int(pred[0])
    py = int(pred[1])
    return _bounded_pixel(px, py, width, height)


def _bounded_pixel(px, py, width, height):
    if px < 0 or py < 0 or px >= width or py >= height:
        return None
    return px, py


def _normalized_offset(value, extent):
    if extent <= 0:
        return 0
    return _bounded_coordinate(int(float(value) * extent), extent)


def _bounded_coordinate(value, extent):
    return max(0, min(int(extent) - 1, int(value)))


def _ellipse_scale(width, height, sample, current_roi):
    if sample.cropped and current_roi:
        _, _, roi_width, roi_height = current_roi
        return roi_width, roi_height
    if sample.cropped:
        return width * (512.0 / 640.0), height * (512.0 / 480.0)
    return width, height
