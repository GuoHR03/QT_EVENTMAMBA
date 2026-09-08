"""Thread-safe lifecycle state machine shared by runtime services."""

from threading import RLock


STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_FAILED = "failed"

# Compatibility name for callers that used the inference-only lifecycle API.
STATE_ERROR = STATE_FAILED

LIFECYCLE_STATES = frozenset(
    (STATE_STOPPED, STATE_STARTING, STATE_RUNNING, STATE_STOPPING, STATE_FAILED)
)

_ALLOWED_TRANSITIONS = {
    STATE_STOPPED: frozenset((STATE_STARTING, STATE_STOPPING, STATE_FAILED)),
    STATE_STARTING: frozenset(
        (STATE_RUNNING, STATE_STOPPING, STATE_STOPPED, STATE_FAILED)
    ),
    STATE_RUNNING: frozenset((STATE_STOPPING, STATE_STOPPED, STATE_FAILED)),
    STATE_STOPPING: frozenset((STATE_STOPPED, STATE_STARTING, STATE_FAILED)),
    STATE_FAILED: frozenset((STATE_STARTING, STATE_STOPPING, STATE_STOPPED)),
}


def validate_lifecycle_state(state):
    if state not in LIFECYCLE_STATES:
        raise ValueError(f"Unknown lifecycle state: {state}")
    return state


def validate_lifecycle_transition(current, target):
    validate_lifecycle_state(current)
    validate_lifecycle_state(target)
    if current != target and target not in _ALLOWED_TRANSITIONS[current]:
        raise RuntimeError(f"Invalid lifecycle transition: {current} -> {target}")
    return target


class LifecycleStateMachine:
    """Own a service state, failure detail, and validated atomic transitions."""

    def __init__(self, initial=STATE_STOPPED, callback=None, lock=None):
        validate_lifecycle_state(initial)
        self._state = initial
        self._last_error = None
        self._callback = callback
        self._lock = lock or RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    def transition(self, target, error=None):
        return self.transition_if(target, lambda: True, error=error)

    def transition_if(self, target, predicate, error=None, on_transition=None):
        """Transition atomically only when ``predicate`` still holds."""
        with self._lock:
            if not predicate():
                return False
            current = self._state
            validate_lifecycle_transition(current, target)
            next_error = str(error) if error else None
            changed = current != target or self._last_error != next_error
            self._state = target
            self._last_error = next_error
            if on_transition is not None:
                on_transition()
            callback = self._callback if changed else None

        if callback is not None:
            try:
                callback(target, next_error)
            except Exception:
                # Reporting must never prevent cleanup or another transition.
                pass
        return True
