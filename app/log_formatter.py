try:
    from .prediction_overlay import format_prediction_log
except ImportError:
    from prediction_overlay import format_prediction_log


MODE_DISPLAY_NAMES = {
    "center": "中心点",
    "ellipse": "椭圆",
}

NOISE_FILTER_DISPLAY_NAMES = {
    "none": "None",
    "activity": "Activity",
    "trail": "Trail",
    "stc": "STC",
    "anti_flicker": "AntiFlicker",
}


def mode_display_name(mode):
    return MODE_DISPLAY_NAMES.get(mode, mode)


def backend_message(message):
    return format_prediction_log(message, MODE_DISPLAY_NAMES)


def noise_settings_message(filter_type, threshold_us):
    filter_name = NOISE_FILTER_DISPLAY_NAMES.get(filter_type, filter_type)
    return f"去噪设置已更新：{filter_name}, threshold={threshold_us}us"


def roi_settings_message(roi, mode):
    x, y, width, height = roi
    return (
        f"感兴趣区域已设置：x={x}, y={y}, 宽度={width}, "
        f"高度={height}, 模式={mode_display_name(mode)}"
    )
