from .display_names import MODE_DISPLAY_NAMES, NOISE_FILTER_DISPLAY_NAMES
from .prediction_overlay import format_prediction_log


def mode_display_name(mode):
    return MODE_DISPLAY_NAMES.get(mode, mode)


def backend_message(message):
    return format_prediction_log(message, MODE_DISPLAY_NAMES)


def noise_settings_message(filter_type, threshold_us):
    filter_name = NOISE_FILTER_DISPLAY_NAMES.get(filter_type, filter_type)
    return f"去噪设置已更新：{filter_name}, threshold={threshold_us}us"


def roi_settings_message(roi, mode):
    if not roi:
        return f"感兴趣区域已清除，模式={mode_display_name(mode)}"
    x, y, width, height = roi
    return (
        f"感兴趣区域已设置：x={x}, y={y}, 宽度={width}, "
        f"高度={height}, 模式={mode_display_name(mode)}"
    )
