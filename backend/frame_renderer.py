from abc import ABC, abstractmethod


class FrameRenderer(ABC):
    """Common display renderer interface for event sources."""

    @abstractmethod
    def process_events(self, events):
        raise NotImplementedError

    @abstractmethod
    def set_display_settings(self, palette_type=None, fps=None):
        raise NotImplementedError

    def reset(self):
        return False

    def close(self):
        return None
