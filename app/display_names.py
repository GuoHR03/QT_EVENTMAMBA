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

NOISE_FILTER_VALUES_BY_DISPLAY = {
    display_name: value
    for value, display_name in NOISE_FILTER_DISPLAY_NAMES.items()
}
