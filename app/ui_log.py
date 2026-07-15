LOG_LEVEL_KEYWORDS = {
    "error": (
        "error",
        "failed",
        "failure",
        "exception",
        "traceback",
        "失败",
        "错误",
        "异常",
    ),
    "warning": (
        "warning",
        "warn",
        "unsupported",
        "timeout",
        "missing",
        "警告",
        "超时",
        "缺少",
    ),
    "success": (
        "success",
        "ready",
        "loaded",
        "completed",
        "成功",
        "就绪",
        "已加载",
        "完成",
    ),
}


def log_level_for_message(message):
    text = str(message or "").lower()
    for level in ("error", "warning", "success"):
        if any(keyword in text for keyword in LOG_LEVEL_KEYWORDS[level]):
            return level
    if text.startswith(("[noisefilter]", "[roi]")):
        return "info"
    return "default"
