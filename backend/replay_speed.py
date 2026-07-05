from threading import Lock

from backend.settings import DEFAULT_REPLAY_FACTOR


def normalize_replay_factor(replay_factor):
    return max(float(replay_factor or DEFAULT_REPLAY_FACTOR), 0.001)


class ReplaySpeedController:
    def __init__(self, replay_factor=DEFAULT_REPLAY_FACTOR):
        self._lock = Lock()
        self._replay_factor = normalize_replay_factor(replay_factor)

    def get(self):
        with self._lock:
            return self._replay_factor

    def set(self, replay_factor):
        replay_factor = normalize_replay_factor(replay_factor)
        with self._lock:
            self._replay_factor = replay_factor
        return replay_factor
