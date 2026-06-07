"""
ingest.py — Build ChromaDB collections from ielts_relabeled_v3.csv.

Run once (or re-run to rebuild):
    python src/services/ingest.py

Creates two collections:
    essay_col    — embedded on essay text
    question_col — embedded on question text

Metadata stored per document:
    band_bin, task_type, overall, tr, cc, lr, ra, examiner_comment
"""

import os
import sys
import pandas as pd
import chromadb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/ielts_relabeled_v3.csv"
CHROMA_PATH = "data/chroma"

ESSAY_COLLECTION    = "essay_col"
QUESTION_COLLECTION = "question_col"

LABEL_COLS = ["Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]
# ─────────────────────────────────────────────────────────────────────────────


def band_bin(overall: float) -> str:
    """Bin overall score into 4 categories for metadata filtering."""
    if overall <= 4.5:
        return "poor"
    elif overall <= 5.5:
        return "developing"
    elif overall <= 7.0:
        return "competent"
    else:
        return "expert"


def build_collections():
    df = pd.read_csv(DATA_PATH)

    # drop rows with missing labels
    df = df.dropna(subset=LABEL_COLS + ["Examiner_Comment"])
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} labeled essays from {DATA_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # delete and recreate to allow clean rebuild
    for name in [ESSAY_COLLECTION, QUESTION_COLLECTION]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    essay_col    = client.create_collection(ESSAY_COLLECTION)
    question_col = client.create_collection(QUESTION_COLLECTION)

    ids       = []
    essays    = []
    questions = []
    metadatas = []

    for i, row in df.iterrows():
        doc_id = f"essay_{i}"
        ids.append(doc_id)
        essays.append(str(row["Essay"]))
        questions.append(str(row["Question"]))
        metadatas.append({
            "band_bin":         band_bin(float(row["Overall"])),
            "task_type":        str(row.get("Task_Type", "")).strip(),
            "overall":          float(row["Overall"]),
            "tr":               float(row["Task_Response"]),
            "cc":               float(row["Coherence_Cohesion"]),
            "lr":               float(row["Lexical_Resource"]),
            "ra":               float(row["Range_Accuracy"]),
            "examiner_comment": str(row["Examiner_Comment"]),
        })

    # upsert in batches of 500 (chroma default limit)
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = min(start + BATCH, len(ids))
        essay_col.add(
            ids=ids[start:end],
            documents=essays[start:end],
            metadatas=metadatas[start:end],
        )
        question_col.add(
            ids=ids[start:end],
            documents=questions[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Upserted {end}/{len(ids)}")

    print(f"\nDone.")
    print(f"  {ESSAY_COLLECTION}:    {essay_col.count()} documents")
    print(f"  {QUESTION_COLLECTION}: {question_col.count()} documents")
    print(f"  Persisted to {CHROMA_PATH}")


if __name__ == "__main__":
    build_collections()