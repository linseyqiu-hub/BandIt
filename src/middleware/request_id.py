"""
middleware/request_id.py — request ID + one start/done log line per request.

Jobs:
  INWARD  (for logs)  put the ID in this request's context (request_id_var), so
                      RequestIdFilter stamps it on every log line of this request.
  OUTWARD (for users) add an `x-request-id` response header, so a user can report
                      an error ID and we can grep the exact trace.
  TRACE               log "request start" and "request done" (method, path,
                      status, duration_ms). Replaces uvicorn's access log, which
                      is turned off with --no-access-log in the Dockerfile.

Order of events for one request (code order != run order):

    ① uvicorn calls __call__            ② make ID, put it in the context
    ③ log "request start"               ④ await self.app(...)  → FastAPI runs
    ⑤ FastAPI calls send_with_header(response.start)  ← status read here
    ⑥ FastAPI calls send_with_header(response.body)
    ⑦ FastAPI returns to us             ⑧ finally: log "request done", reset context

    send_with_header is DEFINED before ④ but RUNS at ⑤/⑥: it is a set of
    instructions handed to FastAPI, which calls it when it sends the response.

Plain ASGI middleware, not Starlette's BaseHTTPMiddleware: BaseHTTPMiddleware has
known problems with contextvars and streaming responses.

Must be registered LAST in create_app() so it is the outermost of OUR middlewares.
FastAPI still wraps one layer around it: ServerErrorMiddleware, which turns an
unhandled exception into a 500 OUTSIDE this layer. That 500 never passes through
send_with_header, so the except branch records status=500 itself (and that one
response carries no x-request-id header).

The ID is always generated here, never taken from the incoming request: a
client-supplied value would be user-controlled text in every log line.
"""

import logging
import time

from core.request_context import new_request_id, request_id_var

logger = logging.getLogger(__name__)

HEADER_NAME = b"x-request-id"   # ASGI header names/values are bytes

# Still get an ID and the header, but no start/done lines:
#   /health — success is noise; a hung process can't log its own failure (Docker records it)
#   /ready  — logs its own failures, with the reason (routers/health.py)
SKIP_PATHS = {"/health", "/ready"}


class RequestIdMiddleware:
    def __init__(self, app):
        self.app = app                          # the next layer inside

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":             # lifespan / websocket: pass through
            await self.app(scope, receive, send)
            return

        # ② ID first, before any other code runs
        request_id = new_request_id()
        token = request_id_var.set(request_id)  # token remembers the previous value

        method = scope["method"]
        path = scope["path"]                    # path only, never the query string (privacy)
        log = path not in SKIP_PATHS
        status = None                           # unknown until FastAPI sends the response
        crash_error = None                      # exception type, only if one flew past us

        # Defined now, RUN later (⑤/⑥) — FastAPI calls it to send its response.
        async def send_with_header(message):
            nonlocal status                     # write the OUTER variable, not a new local one
            if message["type"] == "http.response.start":
                status = message["status"]      # FastAPI already decided it; we just read it
                message["headers"] = list(message.get("headers", [])) + [
                    (HEADER_NAME, request_id.encode("ascii")),   # str -> bytes, not encryption
                ]
            await send(message)                 # hand on to the real send

        # ③ %r: the path is user-controlled; quoting escapes newlines (no forged log lines)
        if log:
            logger.info("request start method=%s path=%r", method, path)
        start = time.perf_counter()

        try:
            await self.app(scope, receive, send_with_header)    # ④ … ⑦
        except Exception as e:
            status = 500                        # crash: the 500 is built outside this layer
            crash_error = type(e).__name__      # name it in OUR line, next to the request ID
            raise                               # never swallow; uvicorn prints the traceback
        finally:
            # ⑧
            if log:
                duration_ms = (time.perf_counter() - start) * 1000
                level = logging.ERROR if status is None or status >= 500 else logging.INFO
                if crash_error is None:
                    logger.log(
                        level,
                        "request done method=%s path=%r status=%s duration_ms=%d",
                        method, path, status, duration_ms,
                    )
                else:
                    logger.log(
                        level,
                        "request done method=%s path=%r status=%s duration_ms=%d error=%s",
                        method, path, status, duration_ms, crash_error,
                    )
            # Whatever you set, you reset: the context goes back to "-" so nothing
            # logged after this request can show a stale ID.
            request_id_var.reset(token)