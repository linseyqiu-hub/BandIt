"""
BandIt RAG ingest script
Loads ielts_relabeled_v3.csv and populates two ChromaDB collections:
  - essays:    embeddings from Essay text,    documents = Examiner_Comment
  - questions: embeddings from Question text, documents = Examiner_Comment

At query time, the returned document IS the examiner_comment (few-shot payload).
Embeddings are computed explicitly via sentence-transformers (all-MiniLM-L6-v2).

Usage:
    python src/scripts/ingest.py
    python src/scripts/ingest.py --data data/ielts_relabeled_v3.csv --db data/chroma
"""

import os
import sys
import argparse
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA   = os.path.join("data", "ielts_relabeled_v3.csv")
DEFAULT_DB     = os.path.join("data", "chroma")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH           = 500

# ---------------------------------------------------------------------------
# Band bin — same 4-bin stratification used in dataset.py
# ---------------------------------------------------------------------------
def compute_band_bin(overall: float) -> str:
    if overall < 5.0:
        return "poor"
    elif overall < 6.5:
        return "developing"
    elif overall < 8.0:
        return "competent"
    else:
        return "expert"

# ---------------------------------------------------------------------------
# Embed helper
# ---------------------------------------------------------------------------
def embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=True).tolist()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_path: str, db_path: str) -> None:
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

    # --- ChromaDB ---
    print(f"\nInitialising ChromaDB at {db_path} ...")
    # disable chromadb's own embedding function — we provide embeddings explicitly
    client = chromadb.PersistentClient(path=db_path)

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
    parser.add_argument("--db",   default=DEFAULT_DB,   help="ChromaDB persistent directory")
    args = parser.parse_args()

    main(args.data, args.db)