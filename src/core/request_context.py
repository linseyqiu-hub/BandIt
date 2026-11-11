"""
core/request_context.py — per-request values that any code can read.

request_id_var is a ContextVar: ONE label, but a separate value per request.
Each request runs in its own asyncio Task, and each Task has its own Context
(think: a backpack). set() writes into the CURRENT request's backpack; get()
reads from it. `def` routes run in worker threads, and anyio copies the
request's Context into the thread, so the value is visible there too.

    writer : middleware/request_id.py   (sets it once, at the start of a request)
    reader : core/logging_setup.py      (RequestIdFilter stamps it on every log record)

Both import it from here, so neither depends on the other.
"""

import uuid
from contextvars import ContextVar

# "-" = no request (startup, shutdown, background work).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """12 random hex chars. Random, not sequential: reveals nothing, unlocks nothing."""
    return uuid.uuid4().hex[:12]