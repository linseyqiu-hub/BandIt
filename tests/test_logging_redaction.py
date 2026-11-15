"""
Privacy guarantee: exception MESSAGES never reach the logs.

RedactingFormatter must keep what makes a traceback useful (exception type,
function names, files, chained causes) and drop the message, the only part
built from runtime data (and so the only way user text could leak in).
"""

import io
import logging

from core.logging_setup import REDACTED, RedactingFormatter

SECRET = "MY SECRET ESSAY TEXT"


def _log_exception_and_capture(raise_fn) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("test.redaction")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raise_fn()
        except Exception as e:
            logger.exception("failed error=%s", type(e).__name__)
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


def _score_with_user_text():
    raise ValueError(f"could not score essay: {SECRET}")


def test_message_is_redacted_but_traceback_kept():
    out = _log_exception_and_capture(_score_with_user_text)
    assert SECRET not in out
    assert f"ValueError: {REDACTED}" in out
    assert "_score_with_user_text" in out          # the frame (where it broke) is still there
    assert "failed error=ValueError" in out        # our own line is untouched


def _wrapped():
    try:
        _score_with_user_text()
    except ValueError as e:
        raise RuntimeError(f"pipeline failed for {SECRET}") from e


def test_chained_exceptions_are_all_redacted():
    out = _log_exception_and_capture(_wrapped)
    assert SECRET not in out
    assert f"ValueError: {REDACTED}" in out
    assert f"RuntimeError: {REDACTED}" in out
    assert "direct cause" in out
    # Root cause first, like Python's own tracebacks
    assert out.index("ValueError") < out.index("RuntimeError: ")


def test_redaction_survives_another_formatter_running_first():
    # logging caches the formatted traceback on the record (record.exc_text).
    # A plain Formatter running first (pytest's caplog, a future error tracker)
    # would cache the UNREDACTED text; ours must not reuse it.
    try:
        _score_with_user_text()
    except ValueError:
        import sys
        record = logging.getLogger("test.cache").makeRecord(
            "test.cache", logging.ERROR, __file__, 0, "failed", (), sys.exc_info()
        )
    logging.Formatter("%(message)s").format(record)           # caches the unredacted text
    assert SECRET in record.exc_text                           # (proves the cache exists)
    out = RedactingFormatter("%(message)s").format(record)
    assert SECRET not in out
    assert f"ValueError: {REDACTED}" in out
