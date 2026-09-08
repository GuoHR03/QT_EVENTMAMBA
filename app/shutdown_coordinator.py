"""Coordinate ordered, retryable application shutdown phases."""

from threading import RLock

from backend.lifecycle import (
    STATE_FAILED,
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_STOPPING,
    LifecycleStateMachine,
)


SHUTDOWN_IDLE = "idle"
SHUTDOWN_UI_RESOURCES = "ui_resources"
SHUTDOWN_BACKEND = "backend"
SHUTDOWN_FINALIZING = "finalizing"
SHUTDOWN_COMPLETE = "complete"
SHUTDOWN_FAILED = "failed"


class ApplicationShutdownCoordinator:
    """Own shutdown ordering while respecting UI and worker thread affinity."""

    def __init__(self, controller, operation_state, state_callback=None):
        self.controller = controller
        self.operation_state = operation_state
        self._lock = RLock()
        self._lifecycle = LifecycleStateMachine(
            initial=STATE_RUNNING,
            callback=state_callback,
            lock=self._lock,
        )
        self._ui_resources_closed = False
        self._backend_closed = False
        self._errors = []

    @property
    def pending(self):
        return bool(self.operation_state.close_pending)

    @property
    def ready(self):
        return bool(self.operation_state.close_ready)

    @property
    def state(self):
        return self._lifecycle.state

    @property
    def last_error(self):
        return self._lifecycle.last_error

    @property
    def errors(self):
        with self._lock:
            return tuple(self._errors)

    @property
    def phase(self):
        state = self.state
        if state == STATE_FAILED:
            return SHUTDOWN_FAILED
        if state == STATE_STOPPED:
            return SHUTDOWN_COMPLETE
        if state == STATE_RUNNING:
            return SHUTDOWN_IDLE
        with self._lock:
            if not self._ui_resources_closed:
                return SHUTDOWN_UI_RESOURCES
            if not self._backend_closed:
                return SHUTDOWN_BACKEND
            return SHUTDOWN_FINALIZING

    def begin(self):
        """Begin shutdown or resume the first unfinished phase after failure."""
        with self._lock:
            if self.ready or self.pending:
                return False
            if not self.operation_state.begin_close():
                return False
            self._lifecycle.transition(STATE_STOPPING)
            return True

    def close_ui_resources(self):
        """Run on the UI thread; stop camera/playback and Qt network objects."""
        self._require_pending()
        with self._lock:
            if self._ui_resources_closed:
                return True
        try:
            self.controller.close_ui_resources()
        except Exception as exc:
            self._record_failure(SHUTDOWN_UI_RESOURCES, exc)
            raise
        with self._lock:
            self._ui_resources_closed = True
        return True

    def close_backend_resources(self):
        """Run in the shutdown worker after the UI-owned resources are gone."""
        self._require_pending()
        with self._lock:
            if not self._ui_resources_closed:
                raise RuntimeError(
                    "UI resources must close before backend resources"
                )
            if self._backend_closed:
                return True
        try:
            self.controller.close_backend_resources()
        except Exception as exc:
            self._record_failure(SHUTDOWN_BACKEND, exc)
            raise
        with self._lock:
            self._backend_closed = True
        return True

    def complete(self):
        """Commit application close after the backend worker has finished."""
        with self._lock:
            if not self._ui_resources_closed or not self._backend_closed:
                raise RuntimeError("Application shutdown is not complete")
            self._lifecycle.transition(STATE_STOPPED)
            self.operation_state.complete_close()
        return True

    def abort(self, error=None):
        """Re-enable close retries while retaining completed phase markers."""
        with self._lock:
            if self.pending:
                self.operation_state.abort_close()
            if self.state == STATE_STOPPING:
                failure = error or "Application shutdown was interrupted"
                self._record_failure_locked(self.phase, failure)
        return True

    def _require_pending(self):
        if not self.pending:
            raise RuntimeError("Application shutdown has not been requested")

    def _record_failure(self, phase, error):
        with self._lock:
            self._record_failure_locked(phase, error)

    def _record_failure_locked(self, phase, error):
        message = str(error)
        self._errors.append((phase, message))
        self._lifecycle.transition(STATE_FAILED, error=message)

