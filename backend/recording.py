import time


class RawRecorder:
    def __init__(self, clock=None):
        self.clock = clock or time.strftime
        self.is_recording = False

    def start(self, device):
        events_stream = self._events_stream(device)
        if events_stream is None:
            self.is_recording = False
            return False

        timestamp = self.clock("%Y%m%d_%H%M%S")
        events_stream.start_log_raw_data(f"recording_{timestamp}.raw")
        self.is_recording = True
        return True

    def stop(self, device):
        events_stream = self._events_stream(device)
        if events_stream is None:
            self.is_recording = False
            return False

        events_stream.stop_log_raw_data()
        self.is_recording = False
        return True

    @staticmethod
    def _events_stream(device):
        if device is None:
            return None
        return device.get_i_events_stream()
