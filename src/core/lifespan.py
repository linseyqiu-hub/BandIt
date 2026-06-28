"""
BandIt app lifespan
Loads all heavy resources once at startup, stores on app.state,
releases on shutdown.

app.state:
    inference_engine  — BandItInferenceEngine (DeBERTa scorer)
    embedding_model   — SentenceTransformer (MiniLM, for RAG retrieval)
    essays_col        — ChromaDB collection handle (essay embeddings)
    questions_col     — ChromaDB collection handle (question embeddings)
"""

import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from core.config import MODEL_PATH, CHROMA_DB_PATH
from inference import BandItInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------ #
    # STARTUP                                                              #
    # ------------------------------------------------------------------ #

    # 1. scoring model
    print("[lifespan] loading BandIt inference engine...")
    app.state.inference_engine = BandItInferenceEngine(MODEL_PATH)
    print("[lifespan] inference engine ready ✓")

    # 2. embedding model (for RAG retrieval)
    print("[lifespan] loading embedding model (all-MiniLM-L6-v2)...")
    app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[lifespan] embedding model ready ✓")

    # 3. ChromaDB collections
    print(f"[lifespan] connecting to ChromaDB at {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    app.state.essays_col    = chroma_client.get_collection("essays",    embedding_function=None)
    app.state.questions_col = chroma_client.get_collection("questions", embedding_function=None)
    print(f"[lifespan] essays    count: {app.state.essays_col.count()} ✓")
    print(f"[lifespan] questions count: {app.state.questions_col.count()} ✓")

    print("[lifespan] all resources loaded — app ready\n")

    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #
    print("[lifespan] shutting down...")
    # SentenceTransformer and ChromaDB have no explicit close — GC handles it
    # BandItInferenceEngine — same
    print("[lifespan] shutdown complete")