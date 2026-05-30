import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.lifespan import lifespan
from routers import scoring


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
# origins=["*"] is fine for local dev — tighten this before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(scoring.router)

# Future routers added here when ready:
# app.include_router(feedback.router)
# app.include_router(speaking.router)


# ------------------------------------------------------------------
# Health + liveness
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    """
    Returns whether the ML model is loaded and ready.
    Checks app.state.engine — set by lifespan on startup.
    """
    ready = hasattr(app.state, "engine") and app.state.engine is not None
    return {
        "status": "healthy" if ready else "unhealthy",
        "model":  "deberta-v3-base",
        "ready":  ready,
    }