"""
middleware/request_id.py — give every HTTP request an ID.

Two jobs:
  INWARD  (for logs)  put the ID in this request's context (request_id_var), so
                      RequestIdFilter stamps it on every log line of this request.
  OUTWARD (for users) add an `x-request-id` response header, so a user can report
                      an error ID and we can grep the exact trace.

Plain ASGI middleware, not Starlette's BaseHTTPMiddleware: BaseHTTPMiddleware has
known problems with contextvars and streaming responses.

Must be registered LAST in create_app() so it is the OUTERMOST layer: the ID then
exists before any other middleware or route code runs.

The ID is always generated here, never taken from the incoming request: a
client-supplied value would be user-controlled text in every log line.
"""

from core.request_context import new_request_id, request_id_var

HEADER_NAME = b"x-request-id"   # ASGI header names/values are bytes


class RequestIdMiddleware:
    def __init__(self, app):
        self.app = app                          # the next layer inside

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":             # lifespan / websocket: pass through
            await self.app(scope, receive, send)
            return

        request_id = new_request_id()
        token = request_id_var.set(request_id)  # token remembers the previous value

        async def send_with_header(message):
            # The response leaves as messages; the first one carries status + headers.
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (HEADER_NAME, request_id.encode("ascii")),   # str -> bytes, not encryption
                ]
            await send(message)                 # hand on to the real send

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            # Whatever you set, you reset: the context goes back to "-" so nothing
            # logged after this request can show a stale ID.
            request_id_var.reset(token)