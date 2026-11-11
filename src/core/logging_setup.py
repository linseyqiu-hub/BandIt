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

    Uvicorn's loggers are untouched: uvicorn configures them itself, before
    create_app() runs, with propagate=False — so its lines keep uvicorn's own
    format and are never printed twice.

Call configure_logging() first thing in an entrypoint, after settings load.
"""

import logging
import sys
import time

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%SZ"   # "Z" = UTC, see converter below

# Libraries that log at INFO on every call. They get their own level, so
# they stop borrowing root's. Add more only after seeing real noisy output.
NOISY_LOGGERS = {
    "httpx":    logging.WARNING,   # "HTTP Request: POST https://api.anthropic.com/..." per call
    "httpcore": logging.WARNING,   # connection-level detail under httpx
}


def configure_logging(level: str) -> None:
    """
    Attach one stderr handler to the root logger and set the global level.

    Safe to call more than once: existing root handlers are replaced, never
    stacked, so a second call cannot produce duplicate lines.
    """
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    # UTC timestamps: containers already run in UTC, but a host run would use
    # local time. One clock everywhere makes laptop and cloud logs comparable.
    formatter.converter = time.gmtime

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for old in root.handlers[:]:          # copy: we mutate the list while looping
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)

    for name, noisy_level in NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(noisy_level)