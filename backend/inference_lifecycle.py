"""Shared inference lifecycle states without service or Qt dependencies."""


STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_ERROR = "error"

LIFECYCLE_STATES = frozenset(
    (STATE_STOPPED, STATE_STARTING, STATE_RUNNING, STATE_STOPPING, STATE_ERROR)
)


def validate_lifecycle_state(state):
    if state not in LIFECYCLE_STATES:
        raise ValueError(f"Unknown inference lifecycle state: {state}")
    return state
