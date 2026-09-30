"""
BandIt app lifespan
Loads all heavy resources once at startup, stores on app.state,
releases on shutdown.

app.state:
    inference_engine  — BandItInferenceEngine (DeBERTa scorer)
    embedding_model   — SentenceTransformer (MiniLM, for RAG retrieval)
    chroma_client     — ChromaDB HttpClient (vector search)
"""

import os
from contextlib import asynccontextmanager

from chromadb import HttpClient
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from core.config import CHROMA_HOST, CHROMA_PORT
from inference import BandItInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------ #
    # STARTUP                                                              #
    # ------------------------------------------------------------------ #

    # 1. scoring model
    print("[lifespan] loading BandIt inference engine...")
    app.state.inference_engine = BandItInferenceEngine()
    print("[lifespan] inference engine ready ✓")

    # 2. embedding model (for RAG retrieval)
    print("[lifespan] loading embedding model (all-MiniLM-L6-v2)...")
    app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[lifespan] embedding model ready ✓")

    # 3. ChromaDB — retry loop because Chroma may still be starting
    import time
    print(f"[lifespan] connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = None
    for attempt in range(15):
        try:
            client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            break
        except Exception:
            wait = min(2 ** attempt * 0.5, 8)   # 0.5, 1, 2, 4, 8, 8, ...
            print(f"[lifespan] chroma not ready, retry in {wait:.1f}s (attempt {attempt + 1}/15)")
            time.sleep(wait)
    if client is None:
        raise RuntimeError("ChromaDB not reachable after 15 attempts")

    app.state.chroma_client = client
    print(f"[lifespan] chroma connected ✓")

    print("[lifespan] all resources loaded — app ready\n")

    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #
    print("[lifespan] shutting down...")
    # SentenceTransformer and ChromaDB have no explicit close — GC handles it
    # BandItInferenceEngine — same
    print("[lifespan] shutdown complete")