from backend.settings import DEFAULT_REPLAY_FACTOR


def normalize_replay_factor(replay_factor):
    return max(float(replay_factor or DEFAULT_REPLAY_FACTOR), 0.001)
