import threading

import pytest

from backend.inference_service import InferenceService
from backend.lifecycle import (
    STATE_FAILED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    LifecycleStateMachine,
)


def test_lifecycle_state_machine_validates_transitions_and_clears_error():
    transitions = []
    lifecycle = LifecycleStateMachine(
        callback=lambda state, error: transitions.append((state, error))
    )

    lifecycle.transition(STATE_STARTING)
    lifecycle.transition(STATE_FAILED, error=RuntimeError("open failed"))
    lifecycle.transition(STATE_STOPPING)
    lifecycle.transition(STATE_STOPPED)

    assert lifecycle.state == STATE_STOPPED
    assert lifecycle.last_error is None
    assert transitions == [
        (STATE_STARTING, None),
        (STATE_FAILED, "open failed"),
        (STATE_STOPPING, None),
        (STATE_STOPPED, None),
    ]


def test_lifecycle_state_machine_rejects_skipped_start_phase():
    lifecycle = LifecycleStateMachine()

    with pytest.raises(RuntimeError, match="stopped -> running"):
        lifecycle.transition(STATE_RUNNING)

    assert lifecycle.state == STATE_STOPPED


def test_guarded_transition_keeps_cancellation_and_state_atomic():
    lifecycle = LifecycleStateMachine(initial=STATE_STARTING)
    cancelled = threading.Event()
    cancelled.set()

    assert not lifecycle.transition_if(
        STATE_RUNNING,
        lambda: not cancelled.is_set(),
    )
    assert lifecycle.state == STATE_STARTING


def test_inference_service_delegates_state_to_shared_lifecycle():
    transitions = []
    service = InferenceService.__new__(InferenceService)
    service._state_lock = threading.RLock()
    service._lifecycle = LifecycleStateMachine(
        callback=lambda state, error: transitions.append((state, error)),
        lock=service._state_lock,
    )

    service._set_state(STATE_STARTING)
    service._set_state(STATE_RUNNING)

    assert service.state == STATE_RUNNING
    assert service.last_error is None
    assert transitions == [(STATE_STARTING, None), (STATE_RUNNING, None)]

