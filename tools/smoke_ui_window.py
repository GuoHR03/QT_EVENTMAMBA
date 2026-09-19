"""Construct and cooperatively close the real Qt main window offscreen."""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_smoke_test(timeout_ms=15000):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from app.bootstrap import configure_runtime

    configure_runtime(str(PROJECT_ROOT / "main.py"))

    from PyQt6.QtCore import QSettings, QTimer
    from PyQt6.QtWidgets import QApplication

    from app.preferences import PreferenceStore
    from app.widget import MainWindow

    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(True)
    preference_directory = tempfile.TemporaryDirectory()
    preference_settings = QSettings(
        str(Path(preference_directory.name) / "preferences.ini"),
        QSettings.Format.IniFormat,
    )
    window = MainWindow(
        preference_store=PreferenceStore(preference_settings),
    )
    window.show()

    timed_out = [False]

    def fail_on_timeout():
        if window.shutdown.ready:
            return
        timed_out[0] = True
        print(
            "MainWindow close timed out: phase={}, errors={}".format(
                window.shutdown.phase,
                window.shutdown.errors,
            ),
            file=sys.stderr,
            flush=True,
        )
        application.exit(2)

    QTimer.singleShot(50, window.close)
    QTimer.singleShot(max(1, int(timeout_ms)), fail_on_timeout)
    exit_code = application.exec()

    if timed_out[0]:
        return 2
    if exit_code != 0:
        return int(exit_code)
    if not window.shutdown.ready:
        print("MainWindow exited before shutdown completed", file=sys.stderr)
        return 3
    if window.inference_operation_coordinator.busy:
        print("Inference operation worker remained busy after close", file=sys.stderr)
        return 4
    if window.controller.is_camera_running():
        print("Camera remained active after close", file=sys.stderr)
        return 5
    if window.controller.is_inference_running():
        print("Inference backend remained active after close", file=sys.stderr)
        return 6

    preference_directory.cleanup()
    print("Real Qt MainWindow smoke test passed", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    try:
        return run_smoke_test(args.timeout_ms)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
