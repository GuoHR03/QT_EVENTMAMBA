import time


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
    if text.startswith(("[noisefilter]", "[roi]", "[performance]")):
        return "info"
    return "default"


class PredictionLogThrottle:
    """Rate-limit successful prediction logs without hiding errors/status."""

    def __init__(self, interval_s=1.0, clock=None):
        self.interval_s = max(0.0, float(interval_s))
        self._clock = clock or time.monotonic
        self._last_prediction_log_at = None

    def should_log(self, result):
        if not isinstance(result, dict) or result.get("msg_type") != "PREDICTION":
            return True

        now = self._clock()
        last = self._last_prediction_log_at
        if last is not None and now - last < self.interval_s:
            return False
        self._last_prediction_log_at = now
        return True

    def reset(self):
        self._last_prediction_log_at = None
