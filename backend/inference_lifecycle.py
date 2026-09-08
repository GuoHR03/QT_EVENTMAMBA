"""Compatibility exports for the former inference-only lifecycle module."""

from backend.lifecycle import (  # noqa: F401
    LIFECYCLE_STATES,
    STATE_ERROR,
    STATE_FAILED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    LifecycleStateMachine,
    validate_lifecycle_state,
    validate_lifecycle_transition,
)
