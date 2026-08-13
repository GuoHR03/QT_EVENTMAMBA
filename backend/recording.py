import os
import sys
import time
from pathlib import Path


class RawRecorder:
    def __init__(self, clock=None, output_dir=None):
        self.clock = clock or time.strftime
        self.output_dir = output_dir
        self.is_recording = False

    def start(self, device):
        events_stream = self._events_stream(device)
        if events_stream is None:
            self.is_recording = False
            return False

        timestamp = self.clock("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.raw"
        output_dir = self.output_dir
        if output_dir is None and getattr(sys, "frozen", False):
            output_dir = _frozen_record_dir()
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = str(output_dir / filename)
        events_stream.start_log_raw_data(filename)
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


def _frozen_record_dir(environ=None):
    environ = environ if environ is not None else os.environ
    preferred_root = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
    if preferred_root:
        return Path(preferred_root) / "UI_Event" / "record"
    return Path.home() / "UI_Event" / "record"
