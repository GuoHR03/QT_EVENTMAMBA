from PyQt6.QtCore import QThread, pyqtSignal


class InferenceOperationThread(QThread):
    """Run one blocking inference lifecycle operation away from the UI thread."""

    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)

    def __init__(self, operation_name, operation, parent=None):
        super().__init__(parent)
        self.operation_name = str(operation_name)
        self.operation = operation

    def run(self):
        if self.isInterruptionRequested():
            self.cancelled.emit(self.operation_name)
            return
        try:
            self.operation()
        except Exception as exc:
            self.failed.emit(self.operation_name, str(exc))
            return
        # A close request can arrive while the blocking backend call is in
        # progress. Re-check before publishing success so the UI never starts
        # a new NetworkThread after shutdown has begun.
        if self.isInterruptionRequested():
            self.cancelled.emit(self.operation_name)
            return
        self.succeeded.emit(self.operation_name)
