import pytest

from backend.inference_transport import stop_network_thread


class RetryableNetworkThread:
    cooperative_stop_timeout_ms = 25

    def __init__(self):
        self.running = True
        self.wait_results = [False, True]
        self.stop_count = 0
        self.wait_timeouts = []
        self.interruption_count = 0
        self.deleted = False

    def stop(self):
        self.stop_count += 1

    def requestInterruption(self):
        self.interruption_count += 1

    def isRunning(self):
        return self.running

    def wait(self, timeout_ms):
        self.wait_timeouts.append(timeout_ms)
        stopped = self.wait_results.pop(0)
        if stopped:
            self.running = False
        return stopped

    def terminate(self):
        raise AssertionError("cooperative shutdown must not terminate QThread")

    def deleteLater(self):
        self.deleted = True


def test_network_stop_timeout_retains_handle_for_safe_retry():
    thread = RetryableNetworkThread()

    with pytest.raises(RuntimeError, match="did not stop cooperatively"):
        stop_network_thread(thread)

    assert thread.isRunning()
    assert not thread.deleted

    assert stop_network_thread(thread)
    assert not thread.isRunning()
    assert thread.deleted
    assert thread.stop_count == 2
    assert thread.interruption_count == 2
    assert thread.wait_timeouts == [25, 25]
