"""
BandIt health router — liveness and readiness probes.

Two endpoints, two different questions, two different consequences:

  /health  (liveness)  — "is this process alive?"
                         Cheap, no dependencies, no I/O.
                         Failing means the process is wedged → restart it.

  /ready   (readiness) — "can this process serve a request right now?"
                         Checks loaded models + reachable dependencies.
                         Failing means stop routing traffic here, but do
                         NOT restart — restarting the API cannot fix Chroma.
Logging:
  /health  never logged — success is noise, and a hung process cannot log its
           own failure; Docker records failed probes (docker ps / inspect).
  /ready   logged ONLY when it fails, with which checks failed and the error
           TYPE (never the message). The request middleware skips both paths.

The HTTP STATUS CODE is the contract. Docker HEALTHCHECK, ALB target
groups and Kubernetes probes read the status code and never parse the
body. The body exists for humans debugging with curl.
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.constants import MODEL_NAME

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {"status": "ok"}


@router.get("/health")
async def health():
    """
    Liveness probe. Returns 200 if the process can respond at all.

    Deliberately does not check models or dependencies: a failure here
    should mean "restart this container", and restarting never fixes a
    downstream service.
    """
    return {"status": "alive"}


@router.get("/ready")
def ready(request: Request):
    """
    Readiness probe. Returns 200 only if every resource needed to serve
    a scoring or feedback request is available; 503 otherwise.

    Reports the result of EVERY check rather than short-circuiting on the
    first failure, so one curl tells you the whole picture.

    Plain `def` (not `async def`) on purpose: heartbeat() is a blocking
    network call. FastAPI runs sync endpoints in a worker thread, so a slow
    Chroma can't freeze the event loop for every other request.
    """
    state = request.app.state

    checks = {
        "model":      getattr(state, "inference_engine", None) is not None,
        "embeddings": getattr(state, "embedding_model", None) is not None,
        "llm":        getattr(state, "llm_client", None) is not None,
    }

    # Dependency checks: any failure is a failed check, not a 500 from this
    # endpoint. The error TYPE is kept for the log (generic, so a future
    # redis check is one more entry here, not a new log line).
    errors: dict[str, str] = {}
 
    # Chroma lives in another container.
    try:
        state.chroma_client.heartbeat()
        checks["chroma"] = True
    except Exception as exc:
        checks["chroma"] = False
        errors["chroma"] = type(exc).__name__

    ok = all(checks.values())
    if not ok:
        failed = [name for name, passed in checks.items() if not passed]
        logger.warning(
            "not ready failed_checks=%s errors=%s",
            ",".join(failed),
            ",".join(f"{name}:{err}" for name, err in errors.items()) or "-",
        )
 

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "model":  MODEL_NAME,
            "checks": checks,
        },
    )