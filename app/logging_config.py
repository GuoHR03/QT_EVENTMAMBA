import logging
import os
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import user_data_root


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
UI_LOG_FILENAME = "ui.log"
CRASH_LOG_FILENAME = "crash.log"
_HANDLER_MARKER = "_ui_event_file_handler"


def configure_application_logging(log_dir=None, logger=None):
    """Configure one bounded UTF-8 application log and return its path."""
    logger = logger or logging.getLogger()
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return Path(handler.baseFilename)

    target_dir = _resolve_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / UI_LOG_FILENAME
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return log_path


def write_crash_report(exc_type, value, tb, log_dir=None):
    """Append an unhandled exception report and return the crash-log path."""
    target_dir = _resolve_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    crash_path = target_dir / CRASH_LOG_FILENAME
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    report = "".join(traceback.format_exception(exc_type, value, tb))
    with crash_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{timestamp}] Unhandled exception\n")
        handle.write(report)
        if not report.endswith("\n"):
            handle.write("\n")
        handle.write("\n")
    logging.getLogger(__name__).critical(
        "Unhandled exception; crash report: %s",
        crash_path,
        exc_info=(exc_type, value, tb),
    )
    return crash_path


def _resolve_log_dir(log_dir):
    if log_dir is not None:
        return Path(log_dir)
    configured = os.environ.get("UI_EVENT_LOG_DIR")
    if configured:
        return Path(configured)
    return user_data_root()
