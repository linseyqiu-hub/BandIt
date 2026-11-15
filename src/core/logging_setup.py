"""
core/logging_setup.py — configure Python logging ONCE per process.

Two layers, each with one job:

    Python (this file)   WHAT is logged and HOW it looks → written to stderr.
    Platform             WHERE stderr goes and HOW LONG it is kept:
                           compose  → json-file driver + rotation (compose.yaml)
                           AWS      → awslogs driver → CloudWatch (retention)

So this file never opens a log file and never rotates anything.

How it fits the logging tree:
    Every module does   logger = logging.getLogger(__name__)
    Those loggers have no handlers and no level of their own (NOTSET), so
    every record climbs to the ROOT logger. We put the ONE handler there and
    set the ONE level there; every module inherits both.

    Request IDs: RequestIdFilter copies request_id_var (core/request_context.py)
    onto every record, and the format prints it as [%(request_id)s]. The filter
    sits on the HANDLER, not the root logger: logger filters only run on the
    logger that created the record, never during the climb to root, while
    handler filters run for every record the handler receives.

    Uvicorn: it configures its own loggers before create_app() runs, with
    propagate=False. We point its top logger ("uvicorn") at OUR handler, so
    its lines (startup, crash tracebacks) get our format and our redaction.
    propagate stays False, so nothing is printed twice.

Exceptions: RedactingFormatter keeps the full traceback (files, lines,
functions, exception types, chained causes) but replaces every exception
MESSAGE with "<message redacted>". The message is the only part of a
traceback built from runtime data, so it is the only way user text could
reach the logs. Fail closed: useful details are logged as explicit safe
fields instead (e.g. status_code=429).

Call configure_logging() first thing in an entrypoint, after settings load.
"""

import logging
import sys
import time
import traceback

from core.request_context import request_id_var

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%SZ"   # "Z" = UTC, see converter below

# Libraries that log at INFO on every call. They get their own level, so
# they stop borrowing root's. Add more only after seeing real noisy output.
NOISY_LOGGERS = {
    "httpx":    logging.WARNING,   # "HTTP Request: POST https://api.anthropic.com/..." per call
    "httpcore": logging.WARNING,   # connection-level detail under httpx
}


REDACTED = "<message redacted>"


class RedactingFormatter(logging.Formatter):
    """A Formatter whose tracebacks never contain exception messages."""

    def format(self, record: logging.LogRecord) -> str:
        # logging caches the formatted traceback on the record (record.exc_text)
        # and every later formatter REUSES that cache. If another handler
        # (pytest's caplog, a future error tracker) formats the record first,
        # its unredacted traceback would be reused here. So never trust the
        # cache: always build our own, then restore what was there.
        cached = record.exc_text
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_text = cached

    def formatException(self, exc_info) -> str:
        # Walk the chain from the exception we caught down to its root cause.
        chain = []
        te = traceback.TracebackException(*exc_info)
        while te is not None:
            if te.__cause__ is not None:
                link, nxt = "direct cause", te.__cause__
            elif te.__context__ is not None and not te.__suppress_context__:
                link, nxt = "context", te.__context__
            else:
                link, nxt = None, None
            chain.append((te, link))
            te = nxt

        # Print root cause first, the same order Python itself uses.
        blocks = []
        for te, link in reversed(chain):
            if link == "direct cause":
                blocks.append("\nThe above exception was the direct cause of the following exception:\n")
            elif link == "context":
                blocks.append("\nDuring handling of the above exception, another exception occurred:\n")
            blocks.append(
                "Traceback (most recent call last):\n"
                + "".join(te.stack.format())          # file, line, function, source line
                + f"{te.exc_type_str}: {REDACTED}"     # type kept, message dropped
            )
        return "\n".join(blocks)


class RequestIdFilter(logging.Filter):
    """Stamp the current request's ID on the record. Never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()   # "-" outside a request
        return True


def configure_logging(level: str) -> None:
    """
    Attach one stderr handler to the root logger and set the global level.

    Safe to call more than once: existing root handlers are replaced, never
    stacked, so a second call cannot produce duplicate lines.
    """
    formatter = RedactingFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    # UTC timestamps: containers already run in UTC, but a host run would use
    # local time. One clock everywhere makes laptop and cloud logs comparable.
    formatter.converter = time.gmtime

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())  # on the handler: runs for every record

    root = logging.getLogger()
    for old in root.handlers[:]:          # copy: we mutate the list while looping
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn's own lines (startup, "Exception in ASGI application" tracebacks)
    # through OUR handler too: same format, same redaction. propagate stays
    # False, so they are not printed a second time via root.
    logging.getLogger("uvicorn").handlers = [handler]

    for name, noisy_level in NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(noisy_level)