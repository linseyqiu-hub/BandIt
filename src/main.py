import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import MODEL_NAME, CORS_ORIGINS
from core.lifespan import lifespan
from routers import scoring, feedback


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

app = FastAPI(
    title       = "BandIt API",
    description = "IELTS essay scoring powered by DeBERTa-v3-base.",
    version     = "1.0.0",
    lifespan    = lifespan,   # startup/shutdown handled in core/lifespan.py
)


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

# Allows the React frontend (Week 3) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(scoring.router)
app.include_router(feedback.router)

# Future routers added here when ready:
# app.include_router(feedback.router)
# app.include_router(speaking.router)


# ------------------------------------------------------------------
# Liveness + readiness
# ------------------------------------------------------------------
#
# Two endpoints, two different questions, two different consequences:
#
#   /health  (liveness)  — "is this process alive?"
#                          Cheap, no dependencies, no I/O.
#                          Failing means the process is wedged → restart it.
#
#   /ready   (readiness) — "can this process serve a request right now?"
#                          Checks loaded models + reachable dependencies.
#                          Failing means stop routing traffic here, but do
#                          NOT restart — restarting the API cannot fix Chroma.
#
# The HTTP STATUS CODE is the contract. Docker HEALTHCHECK, ALB target
# groups and Kubernetes probes read the status code and never parse the
# body. The body exists for humans debugging with curl.


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    """
    Liveness probe. Returns 200 if the process can respond at all.

    Deliberately does not check models or dependencies: a failure here
    should mean "restart this container", and restarting never fixes a
    downstream service.
    """
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    """
    Readiness probe. Returns 200 only if every resource needed to serve
    a scoring or feedback request is available; 503 otherwise.

    Reports the result of EVERY check rather than short-circuiting on the
    first failure, so one curl tells you the whole picture.
    """
    checks = {
        "model":      getattr(app.state, "inference_engine", None) is not None,
        "embeddings": getattr(app.state, "embedding_model",  None) is not None,
    }

    # Chroma lives in another container; any failure to reach it is a
    # failed check, not a 500 from this endpoint.
    try:
        app.state.chroma_client.heartbeat()
        checks["chroma"] = True
    except Exception:
        checks["chroma"] = False

    ok = all(checks.values())

    return JSONResponse(
        status_code = 200 if ok else 503,
        content     = {
            "status": "ready" if ok else "not_ready",
            "model":  MODEL_NAME,
            "checks": checks,
        },
    )