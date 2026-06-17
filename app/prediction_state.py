try:
    from .prediction_overlay import parse_prediction_result, select_prediction_for_frame
except ImportError:
    from prediction_overlay import parse_prediction_result, select_prediction_for_frame


class PredictionState:
    def __init__(self, interval_ms=20):
        self.interval_ms = interval_ms
        self.buffer = {}
        self.latest_values = None

    def clear(self):
        self.buffer.clear()
        self.latest_values = None

    def add_result(self, result, timestamp, mode):
        prediction = parse_prediction_result(result, mode)
        if prediction is None:
            return None
        self.buffer[timestamp] = prediction
        self.latest_values = prediction.values
        return prediction

    def match_frame(self, timestamp):
        return select_prediction_for_frame(
            self.buffer,
            timestamp,
            self.interval_ms,
        )
