import sys
import traceback

from app.bootstrap import configure_runtime
from app.logging_config import configure_application_logging, write_crash_report
from app.runtime_smoke import runtime_smoke_exit_code


def exception_hook(exctype, value, tb):
    crash_path = None
    try:
        crash_path = write_crash_report(exctype, value, tb)
    except Exception:
        traceback.print_exception(exctype, value, tb)

    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        application = QApplication.instance()
        if application is None:
            traceback.print_exception(exctype, value, tb)
            return
        details = f"\n\n崩溃日志：{crash_path}" if crash_path else ""
        QMessageBox.critical(
            application.activeWindow(),
            "UI_Event 发生错误",
            "程序遇到未处理的错误，需要退出。" + details,
        )
        application.exit(1)
    except Exception:
        traceback.print_exception(exctype, value, tb)


def main():
    smoke_exit_code = runtime_smoke_exit_code()
    if smoke_exit_code is not None:
        raise SystemExit(smoke_exit_code)

    configure_runtime(__file__)
    try:
        configure_application_logging()
    except Exception:
        # A read-only or invalid log directory must not prevent the UI from
        # starting. The exception hook retains a stderr fallback.
        traceback.print_exc()
    sys.excepthook = exception_hook

    from PyQt6.QtWidgets import QApplication

    from app.widget import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
