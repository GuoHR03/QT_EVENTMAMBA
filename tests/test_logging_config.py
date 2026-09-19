import logging

from app.logging_config import configure_application_logging, write_crash_report


def test_application_logging_writes_utf8_and_does_not_duplicate_handler(tmp_path):
    logger = logging.getLogger("ui_event_test_logging")
    logger.handlers.clear()
    logger.propagate = False

    first_path = configure_application_logging(tmp_path, logger=logger)
    second_path = configure_application_logging(tmp_path, logger=logger)
    logger.info("中文日志")
    for handler in logger.handlers:
        handler.flush()

    assert first_path == second_path
    assert len(logger.handlers) == 1
    assert "中文日志" in first_path.read_text(encoding="utf-8")
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_crash_report_appends_traceback(tmp_path):
    try:
        raise RuntimeError("测试崩溃")
    except RuntimeError as exc:
        crash_path = write_crash_report(type(exc), exc, exc.__traceback__, tmp_path)

    report = crash_path.read_text(encoding="utf-8")
    assert "RuntimeError: 测试崩溃" in report
    assert "Unhandled exception" in report
