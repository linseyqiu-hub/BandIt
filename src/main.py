"""
BandIt API — application factory.

Start with:
    uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000

Startup happens in two phases:

    1. create_app()  — CONFIGURATION: "what to connect to"
                       Loads settings (fails fast, before anything heavy),
                       wires middleware, routers and app.state.settings.
    2. lifespan()    — RESOURCES: "actually connect / load"
                       Reads app.state.settings and builds the model, embedder,
                       Chroma client and LLM client (see core/lifespan.py).

Settings must exist before lifespan runs: Starlette builds the middleware
stack before lifespan starts, so CORS can only be configured here.
"""

import os
import sys
import logging


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.lifespan import lifespan
from core.logging_setup import configure_logging
from core.settings import ServingSettings
from routers import feedback, health, scoring
from middleware.request_id import RequestIdMiddleware

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # ------------------------------------------------------------------
    # Phase 1a — settings. The ONLY place the api process reads its env.
    # Any missing/empty/invalid var raises SettingsError here, listing
    # every problem at once, before a single model weight is loaded.
    # ------------------------------------------------------------------
    settings = ServingSettings.load()

    app = FastAPI(
        title="BandIt API",
        description="IELTS essay scoring powered by DeBERTa-v3-base.",
        version="1.0.0",
        lifespan=lifespan,  # phase 2: resources (core/lifespan.py)
    )
    app.state.settings = settings

    # ------------------------------------------------------------------
    # Phase 1b — logging. Right after settings: the level comes from them.
    # Everything logged from here on (incl. lifespan) is formatted.
    # ------------------------------------------------------------------
    configure_logging(settings.api.log_level)
    logger.info("logging configured level=%s", settings.api.log_level)

    # ------------------------------------------------------------------
    # Phase 1c — middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Browsers hide custom response headers from frontend JS unless exposed.
        expose_headers=["X-Request-ID"],
    )
    # MUST stay last: the last middleware added is the OUTERMOST layer, so the
    # request ID exists before any other middleware or route code runs.
    app.add_middleware(RequestIdMiddleware)

    # ------------------------------------------------------------------
    # Phase 1d — routers
    # ------------------------------------------------------------------
    app.include_router(health.router)    # /, /health, /ready
    app.include_router(scoring.router)   # /score
    app.include_router(feedback.router)  # /feedback
    # future: app.include_router(speaking.router)

    return app