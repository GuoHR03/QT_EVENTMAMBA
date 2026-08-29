"""Pure playback progress and seek state, independent of Qt widgets."""

from dataclasses import dataclass
from typing import Optional


PLAYBACK_SLIDER_MAX = 10000


def format_playback_time_us(timestamp_us):
    timestamp_us = max(0, int(timestamp_us or 0))
    seconds = timestamp_us / 1_000_000.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    second_value = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{second_value:06.3f}"
    return f"{minutes:02d}:{second_value:06.3f}"


@dataclass(frozen=True)
class PlaybackProgressView:
    enabled: bool
    slider_value: int
    label: str
    update_slider: bool = True


@dataclass(frozen=True)
class PlaybackSeek:
    fraction: float
    view: PlaybackProgressView


class PlaybackProgressState:
    def __init__(self, slider_max=PLAYBACK_SLIDER_MAX):
        self.slider_max = max(1, int(slider_max))
        self.total_us = 0
        self.dragging = False

    def update(self, current_us, total_us):
        total_us = max(0, int(total_us or 0))
        current_us = max(0, int(current_us or 0))
        if total_us > 0:
            current_us = min(total_us, current_us)
        self.total_us = total_us
        value = self._value_for_time(current_us)
        return self._view(value, current_us, update_slider=not self.dragging)

    def begin_drag(self):
        if self.total_us <= 0:
            return False
        self.dragging = True
        return True

    def preview(self, value):
        if self.total_us <= 0:
            return None
        value = self._clamp_value(value)
        current_us = self._time_for_value(value)
        return self._view(value, current_us, update_slider=False)

    def finish_drag(self, value):
        self.dragging = False
        if self.total_us <= 0:
            return None
        value = self._clamp_value(value)
        current_us = self._time_for_value(value)
        view = self._view(value, current_us, update_slider=True)
        return PlaybackSeek(value / self.slider_max, view)

    def reset(self):
        self.total_us = 0
        self.dragging = False
        return PlaybackProgressView(
            enabled=False,
            slider_value=0,
            label="--:-- / --:--",
        )

    def _value_for_time(self, current_us):
        if self.total_us <= 0:
            return 0
        return self._clamp_value((current_us / self.total_us) * self.slider_max)

    def _time_for_value(self, value):
        return int((value / self.slider_max) * self.total_us)

    def _view(self, value, current_us, update_slider):
        total_label = (
            format_playback_time_us(self.total_us)
            if self.total_us > 0
            else "--:--"
        )
        return PlaybackProgressView(
            enabled=self.total_us > 0,
            slider_value=value,
            label=f"{format_playback_time_us(current_us)} / {total_label}",
            update_slider=update_slider,
        )

    def _clamp_value(self, value):
        return max(0, min(self.slider_max, int(value)))
