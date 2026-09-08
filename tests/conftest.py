import os
import sys
import types
import warnings


class _BoundSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, *args):
        for callback in tuple(self.callbacks):
            callback(*args)


class _SignalDescriptor:
    def __set_name__(self, _owner, name):
        self.name = name

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        signal_name = f"__test_signal_{self.name}"
        signal = instance.__dict__.get(signal_name)
        if signal is None:
            signal = _BoundSignal()
            instance.__dict__[signal_name] = signal
        return signal


class _QObject:
    def __init__(self, *_args, **_kwargs):
        pass


class _QThread(_QObject):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self._test_running = False
        self._test_interruption_requested = False

    def start(self):
        self._test_running = True

    def isRunning(self):
        return self._test_running

    def requestInterruption(self):
        self._test_interruption_requested = True

    def isInterruptionRequested(self):
        return self._test_interruption_requested


try:
    __import__("PyQt6.QtCore")
except (ImportError, OSError) as exc:
    # Some lightweight test environments have the PyQt6 Python package but
    # not its native Qt DLLs. A stub must now be requested explicitly so a
    # broken local Qt installation cannot silently produce a green test run.
    allow_stubs = os.environ.get("UI_EVENT_ALLOW_QT_STUBS", "").strip().lower()
    if allow_stubs not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "PyQt6.QtCore could not be imported. Run tests from .venv-dev, "
            "or explicitly set UI_EVENT_ALLOW_QT_STUBS=1 for a stub-only "
            "logic test run."
        ) from exc
    warnings.warn(
        "UI_EVENT_ALLOW_QT_STUBS is enabled; Qt thread and signal behavior "
        "is simulated in this test run.",
        RuntimeWarning,
    )
    pyqt6 = sys.modules.get("PyQt6") or types.ModuleType("PyQt6")
    qtcore = types.ModuleType("PyQt6.QtCore")
    qtcore.QObject = _QObject
    qtcore.QThread = _QThread
    qtcore.pyqtSignal = lambda *_args, **_kwargs: _SignalDescriptor()
    pyqt6.QtCore = qtcore
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qtcore
    os.environ["UI_EVENT_QT_MODE"] = "stub"
else:
    os.environ["UI_EVENT_QT_MODE"] = "real"
