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

The HTTP STATUS CODE is the contract. Docker HEALTHCHECK, ALB target
groups and Kubernetes probes read the status code and never parse the
body. The body exists for humans debugging with curl.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.constants import MODEL_NAME

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

    # Chroma lives in another container; any failure to reach it is a
    # failed check, not a 500 from this endpoint.
    try:
        state.chroma_client.heartbeat()
        checks["chroma"] = True
    except Exception:
        checks["chroma"] = False

    ok = all(checks.values())

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "model":  MODEL_NAME,
            "checks": checks,
        },
    )