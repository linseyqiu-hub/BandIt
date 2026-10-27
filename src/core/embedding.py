"""
core/embedding.py — a thread-safe wrapper around SentenceTransformer.

Why this exists
---------------
SentenceTransformer uses a Hugging Face FAST tokenizer (Rust-backed). It keeps
its truncation/padding settings on the shared tokenizer object and rewrites
them when a call asks for different ones. Concurrent calls from FastAPI's
worker threads can therefore collide: Rust refuses overlapping read/write
borrows ("RuntimeError: Already borrowed"), and in the worst case one request
is tokenized with another's settings. This is a known, documented issue.

In today's code every call uses the same settings, so collisions are rare —
but "rare" is not "impossible" (the first concurrent calls on a cold server,
or any future caller/library upgrade that changes settings). Reasoning says a
shared write is possible, so it is guarded by design, not left to testing.

The lock serializes encode() calls process-wide. A short query embeds in a few
milliseconds on CPU, so the cost is negligible next to the multi-second LLM
call that dominates /feedback.

threading.Lock (not asyncio.Lock): callers are worker threads — `def`
endpoints today, asyncio.to_thread after the planned async migration.
"""

import threading

from sentence_transformers import SentenceTransformer


class ThreadSafeEmbedder:
    """Drop-in for SentenceTransformer where only .encode() is used."""

    def __init__(self, model: SentenceTransformer):
        self._model = model
        self._lock = threading.Lock()

    def encode(self, *args, **kwargs):
        with self._lock:
            return self._model.encode(*args, **kwargs)