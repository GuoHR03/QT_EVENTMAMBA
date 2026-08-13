import sys
import traceback

from app.bootstrap import configure_runtime
from app.runtime_smoke import runtime_smoke_exit_code


def exception_hook(exctype, value, tb):
    print("\n========== [!] 捕获到致命崩溃 [!] ==========")
    traceback.print_exception(exctype, value, tb)
    print("==========================================\n")
    try:
        input("程序已崩溃，请查看上方报错信息，然后按回车键退出...")
    except EOFError:
        pass
    raise SystemExit(1)


def main():
    smoke_exit_code = runtime_smoke_exit_code()
    if smoke_exit_code is not None:
        raise SystemExit(smoke_exit_code)

    configure_runtime(__file__)

    from PyQt6.QtWidgets import QApplication

    from app.widget import MainWindow

    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
