from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import CHECKPOINT_PATH
from inference import BandItInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.

    The engine is stored on app.state so every request handler
    can access it via request.app.state.engine — no globals needed.

    Why app.state and not a module-level global?
        Globals make testing hard — you can't easily swap in a
        mock engine. app.state is scoped to one FastAPI instance,
        so tests can create a fresh app with a fake engine.
    """

    # --- Startup ---
    print("[lifespan] starting up ...")

    # BandItInferenceEngine.__init__ loads the model and tokenizer.
    # This takes ~3-5s on first run (tokenizer downloads vocab if not cached).
    # It runs exactly once. All requests share this one loaded engine.
    app.state.engine = BandItInferenceEngine(CHECKPOINT_PATH)

    print("[lifespan] engine ready. serving requests.")

    yield  # server is live here — handles requests until shutdown signal

    # --- Shutdown ---
    # Free the model from memory explicitly.
    # On CPU this is less critical, but good practice.
    print("[lifespan] shutting down ...")
    del app.state.engine
    print("[lifespan] engine released.")