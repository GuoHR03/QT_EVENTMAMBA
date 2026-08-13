import subprocess

import pytest

from backend.backend_process import BackendProcess


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_timeouts = []

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)


def test_backend_process_owns_and_closes_log_handle(monkeypatch, tmp_path):
    fake_process = FakeProcess()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return fake_process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    log_path = tmp_path / "logs" / "backend.log"
    backend = BackendProcess(str(log_path))

    assert backend.start(["backend", "--port", "5555"], str(tmp_path)) is fake_process
    assert backend.is_running()
    assert captured["stdout"] is backend._log_handle

    backend.stop()

    assert fake_process.terminated
    assert fake_process.wait_timeouts == [2]
    assert backend.process is None
    assert backend._log_handle is None


def test_backend_process_closes_log_when_spawn_fails(monkeypatch, tmp_path):
    def fail_popen(*_args, **_kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    backend = BackendProcess(str(tmp_path / "backend.log"))

    with pytest.raises(OSError, match="spawn failed"):
        backend.start(["backend"], str(tmp_path))

    assert backend.process is None
    assert backend._log_handle is None


def test_backend_process_preserves_live_handle_when_stop_fails(monkeypatch, tmp_path):
    class StubbornProcess(FakeProcess):
        def __init__(self):
            super().__init__()
            self.allow_exit = False

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            if not self.allow_exit:
                raise subprocess.TimeoutExpired("backend", timeout)

    process = StubbornProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    backend = BackendProcess(str(tmp_path / "backend.log"))
    backend.start(["backend"], str(tmp_path), instance_nonce="owned-nonce")

    with pytest.raises(RuntimeError, match="did not stop"):
        backend.stop()

    assert backend.process is process
    assert backend._log_handle is not None
    assert not backend._log_handle.closed
    assert backend.instance_nonce == "owned-nonce"

    process.allow_exit = True
    assert backend.stop()
    assert backend.process is None
    assert backend._log_handle is None
