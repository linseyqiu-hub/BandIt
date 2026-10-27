"""
BandIt app lifespan — phase 2 of startup: RESOURCES.

Phase 1 (create_app in main.py) has already loaded and validated settings
onto app.state.settings. This module turns those settings into live,
heavy resources once at startup, stores them on app.state, and releases
them on shutdown. It never reads the environment itself.

Shared-resource contract — read before adding anything to app.state
---------------------------------------------------------------------
Every object stored on app.state is created ONCE per process and shared by
every request. /score and /feedback are plain `def` endpoints, so FastAPI
runs them on a pool of worker threads (default 40), and these objects are
used from many threads AT THE SAME TIME.

Anything added here must therefore be:
  1. THREAD-SAFE for concurrent use — or guarded with a lock, or replaced
     with per-thread instances. Check the library SOURCE for the pinned
     version; do not assume, and do not rely on a concurrency test passing:
     races depend on timing, so a test can show a bug but never prove its
     absence. "We only read it" is not enough: some objects mutate hidden
     internal state when called (Hugging Face fast tokenizers store
     truncation/padding settings on a shared Rust object).
  2. CLOSED ON SHUTDOWN if it holds connections, sockets or files
     (HTTP clients, DB/Redis clients) — add it to the shutdown block.

Current resources on app.state:
  inference_engine  DeBERTa scorer       FAST tokenizer (Rust, transformers 5.x):
                                         guarded by a lock inside the engine around
                                         tokenization only. Forward pass (no_grad,
                                         read-only) runs unlocked. No close.
  embedding_model   MiniLM               FAST tokenizer: wrapped in ThreadSafeEmbedder
                                         (core/embedding.py), which locks encode().
                                         No close.
  chroma_client     Chroma HttpClient    wraps one httpx.Client (thread-safe pool).
                                         No public close; released with the process.
  llm_client        Anthropic            wraps one httpx.Client (thread-safe pool).
                                         Closed on shutdown.
"""

import asyncio
from contextlib import asynccontextmanager

from anthropic import Anthropic
from chromadb import HttpClient
from chromadb.api import ClientAPI
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from core.constants import EMBEDDING_MODEL
from core.embedding import ThreadSafeEmbedder
from inference import BandItInferenceEngine


CHROMA_CONNECT_ATTEMPTS = 15
CHROMA_MAX_BACKOFF_S = 8.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings   # ServingSettings, built by create_app()

    # ------------------------------------------------------------------ #
    # STARTUP                                                              #
    # ------------------------------------------------------------------ #
    # Order: the slow model load runs first, which also gives a freshly
    # started Chroma time to boot — so the retry below usually succeeds on
    # the first attempt. (compose's depends_on: service_healthy is the
    # primary guard; the retry is the safety net.)

    # 1. scoring model — checkpoint source passed in from settings
    print("[lifespan] loading BandIt inference engine...")
    registry = settings.registry
    app.state.inference_engine = BandItInferenceEngine(
        repo=registry.model_repo,
        version=registry.model_version,
        hf_token=registry.hf_token.get_secret_value(),   # unwrapped at point of use
    )
    # Warm-up: one request through the full path, single-threaded, before any
    # traffic — absorbs lazy first-call init so real users don't pay for it.
    app.state.inference_engine.warm_up()
    print("[lifespan] inference engine ready ✓")

    # 2. embedding model — same constant ingest and eval use, so the vectors
    #    we query with match the vectors Chroma was built with
    print(f"[lifespan] loading embedding model ({EMBEDDING_MODEL})...")
    #    Wrapped: its fast tokenizer is not safe to call from several threads.
    app.state.embedding_model = ThreadSafeEmbedder(SentenceTransformer(EMBEDDING_MODEL))
    app.state.embedding_model.encode("warm-up")      # same warm-up idea as above
    print("[lifespan] embedding model ready ✓")

    # 3. vector store
    app.state.chroma_client = await _connect_chroma(
        settings.chroma.host, settings.chroma.port
    )

    # 4. LLM client — ONE per process, reused by every /feedback request.
    #    Reuses pooled TLS connections instead of a fresh handshake per call.
    app.state.llm_client = Anthropic(
        api_key=settings.llm.anthropic_api_key.get_secret_value(),
    )
    print("[lifespan] LLM client ready ✓")

    print("[lifespan] all resources loaded — app ready\n")

    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #
    print("[lifespan] shutting down...")
    app.state.llm_client.close()   # release its HTTP connection pool
    # inference_engine / embedding_model: plain in-memory objects, nothing to close.
    # chroma_client: no public close() in chromadb 1.5.9; freed with the process.
    print("[lifespan] shutdown complete")


async def _connect_chroma(host: str, port: int) -> ClientAPI:
    """
    Connect to Chroma with exponential backoff and return a client that has
    PROVEN it works (heartbeat succeeded).

    A client is only returned after its heartbeat passes, so a half-working
    client from a failed attempt can never leak out. (The previous version
    assigned the client before the heartbeat, so a flapping Chroma could
    leave a stale, unreachable client behind and startup would report
    success.)

    Uses await asyncio.sleep, not time.sleep: lifespan is async, and a
    blocking sleep would freeze the event loop for its whole duration.
    """
    print(f"[lifespan] connecting to ChromaDB at {host}:{port}...")

    for attempt in range(1, CHROMA_CONNECT_ATTEMPTS + 1):
        try:
            candidate = HttpClient(host=host, port=port)
            candidate.heartbeat()
            print("[lifespan] chroma connected ✓")
            return candidate                      # only a proven client leaves
        except Exception as exc:
            if attempt == CHROMA_CONNECT_ATTEMPTS:
                break                             # no pointless sleep after the last try
            wait = min(0.5 * 2 ** (attempt - 1), CHROMA_MAX_BACKOFF_S)   # 0.5, 1, 2, 4, 8, 8, ...
            print(
                f"[lifespan] chroma not ready ({type(exc).__name__}), "
                f"retry in {wait:.1f}s (attempt {attempt}/{CHROMA_CONNECT_ATTEMPTS})"
            )
            await asyncio.sleep(wait)

    raise RuntimeError(
        f"ChromaDB not reachable at {host}:{port} "
        f"after {CHROMA_CONNECT_ATTEMPTS} attempts"
    )