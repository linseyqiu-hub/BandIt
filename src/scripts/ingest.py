"""
BandIt RAG ingest script
Loads ielts_relabeled_v3.csv and populates two ChromaDB collections:
  - essays:    embeddings from Essay text,    documents = Examiner_Comment
  - questions: embeddings from Question text, documents = Examiner_Comment

At query time, the returned document IS the examiner_comment (few-shot payload).
Embeddings are computed explicitly via sentence-transformers (all-MiniLM-L6-v2).

Usage:
    python src/scripts/ingest.py --data data/ielts_relabeled_v3.csv
    python src/scripts/ingest.py --data data/ielts_relabeled_v3.csv --host chroma --port 8000
"""

"""
BandIt RAG ingest script
Loads ielts_relabeled_v3.csv and populates two ChromaDB collections:
  - essays:    embeddings from Essay text,    documents = Examiner_Comment
  - questions: embeddings from Question text, documents = Examiner_Comment

At query time, the returned document IS the examiner_comment (few-shot payload).
Embeddings are computed explicitly via sentence-transformers (all-MiniLM-L6-v2).

Where Chroma is (highest priority first):
    1. --host / --port flags
    2. CHROMA_HOST / CHROMA_PORT env vars (compose / ECS inject these)
    3. env/config.env on the host
    4. ChromaSettings defaults (localhost:8000)
All four are resolved by ChromaSettings — the flags are simply passed in as
constructor arguments, the highest-priority source pydantic-settings has.

This job needs NO secrets: it loads only ChromaSettings.

Usage:
    python src/scripts/ingest.py
    python src/scripts/ingest.py --data data/ielts_relabeled_v3.csv --host localhost --port 8001
"""

import os
import sys
import argparse
import pandas as pd
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer

from core.constants import EMBEDDING_MODEL
from core.scoring_utils import compute_band_bin
from core.settings import ChromaSettings
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA = os.path.join("data", "ielts_relabeled_v3.csv")
BATCH        = 500
# EMBEDDING_MODEL comes from core.constants: the SAME constant serving and
# eval query with. Vectors written here must match the vectors queried there.


# ---------------------------------------------------------------------------
# Embed helper
# ---------------------------------------------------------------------------
def embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=True).tolist()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_path: str, chroma_host: str, chroma_port: int):
    # --- connect FIRST: fail fast ---
    # Embedding the whole dataset takes minutes. Check Chroma is reachable
    # before doing that work, not after.
    print(f"Connecting to ChromaDB at {chroma_host}:{chroma_port} ...")
    client = HttpClient(host=chroma_host, port=chroma_port)
    client.heartbeat()
    print("  ChromaDB reachable ✓\n")

    # --- load ---
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} rows loaded")

    required = {"Question", "Essay", "Examiner_Comment", "Overall"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=list(required))
    dropped = before - len(df)
    if dropped:
        print(f"  WARNING: dropped {dropped} rows with nulls in critical columns")

    df = df.reset_index(drop=True)

    # --- band_bin ---
    df["band_bin"] = df["Overall"].apply(compute_band_bin)
    print(f"  Band distribution:\n{df['band_bin'].value_counts().to_string()}")

    # --- embedding model ---
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # --- compute embeddings explicitly ---
    print("\nEmbedding essays ...")
    essay_embeddings = embed(model, df["Essay"].tolist())

    print("\nEmbedding questions ...")
    question_embeddings = embed(model, df["Question"].tolist())

    # --- ChromaDB: rebuild collections ---
    print("\nRebuilding collections ...")
    for name in ("essays", "questions"):
        try:
            client.delete_collection(name)
            print(f"  Deleted existing collection: {name}")
        except Exception:
            pass

    # embedding_function=None tells ChromaDB we are providing embeddings ourselves
    essays_col    = client.create_collection("essays",    embedding_function=None)
    questions_col = client.create_collection("questions", embedding_function=None)
    print("  Collections created: essays, questions")

    # --- build shared records ---
    ids       = [str(i) for i in df.index]
    documents = df["Examiner_Comment"].tolist()   # what gets returned at query time
    metadatas = [
        {
            "band_bin": row["band_bin"],
            "overall":  float(row["Overall"]),
        }
        for _, row in df.iterrows()
    ]

    # --- upsert essays ---
    total = len(ids)
    print(f"\nUpserting {total} records into 'essays' ...")
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        essays_col.add(
            ids=ids[start:end],
            embeddings=essay_embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  essays: {end}/{total}")

    # --- upsert questions ---
    print(f"\nUpserting {total} records into 'questions' ...")
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        questions_col.add(
            ids=ids[start:end],
            embeddings=question_embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  questions: {end}/{total}")

    # --- verify ---
    print("\nVerification:")
    print(f"  essays    count: {essays_col.count()}")
    print(f"  questions count: {questions_col.count()}")
    print("\nIngest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BandIt ChromaDB ingest")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to ielts_relabeled.csv")
    # default=None: "not given" must be distinguishable from a value, so that
    # an omitted flag falls through to env vars / env/config.env / defaults.
    parser.add_argument("--host", default=None, help="ChromaDB host (overrides CHROMA_HOST)")
    parser.add_argument("--port", type=int, default=None, help="ChromaDB port (overrides CHROMA_PORT)")
    args = parser.parse_args()

    # Only flags actually given become constructor arguments (top priority).
    overrides = {k: v for k, v in {"host": args.host, "port": args.port}.items() if v is not None}
    try:
        chroma = ChromaSettings(**overrides)
    except ValidationError as exc:
        sys.exit(f"invalid Chroma configuration (check CHROMA_HOST / CHROMA_PORT):\n{exc}")

    main(args.data, chroma.host, chroma.port)