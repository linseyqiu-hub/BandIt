"""
eval/build_test_set.py

Builds the held-out RAGAS test set for BandIt.

Pipeline (locked design, see design doc):
    Pass 1 -- verify every row up front, build TRUE availability counts from
              only the clean rows (no estimation, no discount factor)
    Pass 2 -- weighted, order-independent target allocation via water-filling,
              capped against real per-bin availability
    Pass 3 -- sample from the already-clean rows per bin (cannot fail verify
              here, since Pass 1 already filtered)
    Pass 4 -- patch held_out=true onto the selected rows in BOTH ChromaDB
              collections (essays, questions), with a readback check per
              collection. Uses read-merge-write since collection.update()
              REPLACES metadata wholesale rather than merging into it.

Schema notes (per the real ingest.py):
    - CSV columns are capitalized: Question, Essay, Examiner_Comment, Overall,
      Task_Response, Coherence_Cohesion, Lexical_Resource,
      Grammatical_Range_Accuracy
    - `Overall` and the four criterion sub-scores are all raw floats,
      multiples of 0.5 in [1.0, 9.0] -- same validation rule applies to all
      five, since IELTS bands every criterion (and the overall) on the same
      half-point 1-9 scale.
    - `band_bin` is NOT a CSV column -- it's derived from Overall via
      compute_band_bin() into one of: poor / developing / competent / expert.
      Stratification here uses that same derived category, computed the
      same way ingest.py computes it, so eval-set bucketing matches what's
      actually stored as metadata in Chroma.

Outputs:
    eval/test_questions.json
    eval/logs/build_test_set_failures.jsonl
    eval/logs/held_out_patch_failures.jsonl
"""

import json
import os
import random
import sys

import pandas as pd
from pathlib import Path
from common import log_failure, log_fix, count_jsonl, chroma_get_metadata, chroma_update_metadata
SCRIPT_DIR = Path(__file__).resolve().parent  # D:\BandIt\eval
PROJECT_ROOT = SCRIPT_DIR.parent               # D:\BandIt


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.config import CHROMA_DB_PATH  # noqa: E402
from src.core.scoring_utils import VALID_OVERALL_SCORES, validate_overall_score, compute_band_bin  # noqa: E402

import chromadb  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# CSV column name -> output JSON key, for the four criterion sub-scores.
# Kept as an explicit map (rather than a .lower() transform) so a future
# CSV rename doesn't silently break output keys that run_ragas.py relies on.
SUBSCORE_COLUMNS = {
    "Task_Response": "task_response",
    "Coherence_Cohesion": "coherence_cohesion",
    "Lexical_Resource": "lexical_resource",
    "Range_Accuracy": "grammatical_range_accuracy",  # CSV col is Range_Accuracy;
    # JSON key kept as grammatical_range_accuracy to match build_prompt's
    # scores.grammatical_range_accuracy attribute name.
}

CONFIG = {
    "dataset_path": PROJECT_ROOT / "data" / "ielts_relabeled_v3.csv",  # matches ingest.py's DEFAULT_DATA
    "target_n": 45,
    # Weighted toward the two middle categories -- these correspond to the
    # Overall range [5.0, 8.0) where sub-score correlations are strongest,
    # matching the original "weighted toward band 5-8" design intent using
    # the actual categorical scheme ingest.py stores.
    "weighted_categories": {"developing", "competent"},
    "weight_high": 2.0,
    "weight_low": 1.0,
    "min_essay_length": 50,    # matches ESSAY_MIN_WORDS in your scoring config
    "max_essay_length": 1200,  # matches ESSAY_MAX_WORDS -- OPEN QUESTION, see below
    "min_question_chars": 10,  # matches QUESTION_MIN_CHARS
    "random_seed": 42,
    "output_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.json"),
}


# ---------------------------------------------------------------------------
# compute_band_bin() is now imported from core.scoring_utils -- single
# source of truth shared with ingest.py, no local duplicate here anymore.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_dataset(path):
    """
    Loads the relabeled IELTS dataset. Real columns per ingest.py:
    Question, Essay, Examiner_Comment, Overall, plus the four criterion
    sub-score columns (Task_Response, Coherence_Cohesion, Lexical_Resource,
    Grammatical_Range_Accuracy). Adds a synthetic 'id' if the CSV doesn't
    already have one -- ingest.py itself just uses row index as string id,
    so we match that convention for consistency with what's already in
    Chroma.
    """
    df = pd.read_csv(path)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    return df.to_dict("records")


def get_collections():
    """
    Matches ingest.py's client setup exactly: PersistentClient at
    CHROMA_DB_PATH, embedding_function=None on both collections (embeddings
    are supplied explicitly, never computed by Chroma itself).
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    essays_collection = client.get_collection("essays", embedding_function=None)
    questions_collection = client.get_collection("questions", embedding_function=None)
    return essays_collection, questions_collection


# ---------------------------------------------------------------------------
# Pass 1 -- verify every row, build TRUE availability from clean rows only
# ---------------------------------------------------------------------------

def verify_row(row, config):
    """
    Returns (ok: bool, reason: str or None). On success, sets
    row['Overall'] to the normalized float, row['band_bin'] to the derived
    category, and normalizes each of the four criterion sub-score columns
    in place (same validation rule as Overall: multiple of 0.5 in
    [1.0, 9.0]) -- so downstream code always sees clean values for all
    five scores.
    """
    essay = (row.get("Essay") or "").strip()
    question = (row.get("Question") or "").strip()
    comment = (row.get("Examiner_Comment") or "").strip()

    if not essay:
        return False, "empty Essay"
    n_words = len(essay.split())
    if n_words < config["min_essay_length"]:
        return False, f"essay below min length ({config['min_essay_length']} words)"
    if n_words > config["max_essay_length"]:
        return False, f"essay above max length ({config['max_essay_length']} words)"
    if not question:
        return False, "empty Question"
    if len(question) < config["min_question_chars"]:
        return False, f"question below min length ({config['min_question_chars']} chars)"
    if not comment:
        return False, "empty Examiner_Comment"

    ok, normalized = validate_overall_score(row.get("Overall"))
    if not ok:
        return False, f"invalid Overall: {row.get('Overall')!r} (must be multiple of 0.5 in [1.0, 9.0])"
    row["Overall"] = normalized
    row["band_bin"] = compute_band_bin(normalized)

    # Validate + normalize the four criterion sub-scores using the same
    # rule as Overall (IELTS bands every criterion on the same 1-9,
    # half-point scale).
    for csv_col in SUBSCORE_COLUMNS:
        ok, normalized_sub = validate_overall_score(row.get(csv_col))
        if not ok:
            return False, (
                f"invalid {csv_col}: {row.get(csv_col)!r} "
                f"(must be multiple of 0.5 in [1.0, 9.0])"
            )
        row[csv_col] = normalized_sub

    return True, None


def verify_and_bucket(rows, config):
    """
    Single scan: verify every row, log failures, bucket clean rows by
    (derived) band_bin category.
    Returns dict: {band_bin_category: [clean_row, ...]}
    """
    clean_by_bin = {}
    n_total = 0
    n_clean = 0

    for row in rows:
        n_total += 1
        ok, reason = verify_row(row, config)
        if not ok:
            log_failure("build_test_set", row.get("id", "UNKNOWN_ID"), reason, row)
            continue
        n_clean += 1
        clean_by_bin.setdefault(row["band_bin"], []).append(row)

    print(f"[verify] scanned {n_total} rows, {n_clean} clean, "
          f"{n_total - n_clean} logged to build_test_set_failures.jsonl")
    return clean_by_bin


# ---------------------------------------------------------------------------
# Pass 2 -- weighted, order-independent target allocation (water-filling)
# ---------------------------------------------------------------------------

def compute_weights(bins, config):
    return {
        b: config["weight_high"] if b in config["weighted_categories"] else config["weight_low"]
        for b in bins
    }


def allocate_targets(available, config):
    """
    available: dict {band_bin_category: n_clean_rows_available}
    Returns: dict {band_bin_category: capped_target}, logs an
    ALLOCATION_SHORTFALL failure if target_n can't be reached even using
    every clean row.
    """
    bins = list(available.keys())
    weights = compute_weights(bins, config)
    total_weight = sum(weights.values())
    target_n = config["target_n"]

    raw_target = {b: (weights[b] / total_weight) * target_n for b in bins}
    capped_target = {b: min(raw_target[b], available[b]) for b in bins}

    for _ in range(len(bins) + 1):
        allocated = sum(capped_target.values())
        shortfall = target_n - allocated
        if shortfall <= 1e-9:
            break

        room = {b: available[b] - capped_target[b] for b in bins if available[b] > capped_target[b]}
        if not room:
            break

        room_weight_total = sum(weights[b] for b in room)
        for b in room:
            share = (weights[b] / room_weight_total) * shortfall
            capped_target[b] = min(capped_target[b] + share, available[b])

    capped_target = {b: min(int(round(v)), available[b]) for b, v in capped_target.items()}

    final_total = sum(capped_target.values())
    if final_total < target_n:
        log_failure(
            "build_test_set",
            "ALLOCATION_SHORTFALL",
            f"could not reach target_n={target_n} even using all clean rows "
            f"(got {final_total})",
            {"available": available, "capped_target": capped_target},
        )

    print(f"[allocate] target_n={target_n}, achievable={final_total}")
    for b in sorted(capped_target):
        print(f"    {b}: available={available[b]}, target={capped_target[b]}")

    return capped_target


# ---------------------------------------------------------------------------
# Pass 3 -- sample from clean rows (cannot fail verification, already clean)
# ---------------------------------------------------------------------------

def sample_rows(clean_by_bin, capped_target, config):
    rng = random.Random(config["random_seed"])
    selected = []
    for b, target in capped_target.items():
        pool = clean_by_bin.get(b, [])
        rng.shuffle(pool)
        selected.extend(pool[:target])
    return selected


# ---------------------------------------------------------------------------
# Pass 4 -- patch held_out=true in both Chroma collections, read-merge-write
# ---------------------------------------------------------------------------

def patch_held_out(selected_rows, essays_collection, questions_collection):
    collections = {"essays": essays_collection, "questions": questions_collection}
    n_patched = {"essays": 0, "questions": 0}
    n_failed = {"essays": 0, "questions": 0}

    for row in selected_rows:
        row_id = row["id"]
        for coll_name, coll in collections.items():
            chroma_update_metadata(coll, row_id, held_out=True)
            meta = chroma_get_metadata(coll, row_id)
            if meta and meta.get("held_out") is True:
                n_patched[coll_name] += 1
            else:
                n_failed[coll_name] += 1
                log_failure(
                    "held_out_patch",
                    row_id,
                    f"patch not confirmed in {coll_name} collection",
                    {"readback": meta},
                )

    print(f"[patch] essays: {n_patched['essays']} ok / {n_failed['essays']} failed | "
          f"questions: {n_patched['questions']} ok / {n_failed['questions']} failed")
    if n_failed["essays"] or n_failed["questions"]:
        print("  -> see eval/logs/held_out_patch_failures.jsonl before running run_ragas.py")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_test_questions(selected_rows, config):
    out = []
    for row in selected_rows:
        item = {
            "id": row["id"],
            "question": row["Question"],
            "essay": row["Essay"],
            "reference": row["Examiner_Comment"],
            "overall": row["Overall"],
            "band_bin": row["band_bin"],
        }
        for csv_col, json_key in SUBSCORE_COLUMNS.items():
            item[json_key] = row[csv_col]
        out.append(item)
    with open(config["output_path"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[write] {len(out)} rows -> {config['output_path']}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = CONFIG
    rows = load_dataset(config["dataset_path"])

    clean_by_bin = verify_and_bucket(rows, config)
    available = {b: len(rows_) for b, rows_ in clean_by_bin.items()}

    capped_target = allocate_targets(available, config)
    selected_rows = sample_rows(clean_by_bin, capped_target, config)

    write_test_questions(selected_rows, config)

    essays_collection, questions_collection = get_collections()
    patch_held_out(selected_rows, essays_collection, questions_collection)

    print("\n=== summary ===")
    print(f"selected: {len(selected_rows)}")
    print(f"build_test_set failures logged: {count_jsonl('build_test_set_failures.jsonl')}")
    print(f"held_out_patch failures logged: {count_jsonl('held_out_patch_failures.jsonl')}")


if __name__ == "__main__":
    main()