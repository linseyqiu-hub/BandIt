"""
Shared utilities for the BandIt RAGAS evaluation harness.

Provides:
- log_failure(): append a structured failure record for manual review
- log_fix(): append a structured record of an automatic fix that was applied
- read_jsonl(): helper to read logs back for summary printing

Nothing in this module ever raises on a logging failure path silently swallowing
data — if the log write itself fails, that's allowed to raise, since a broken
logger is worse than a crashed script.
"""

import json
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _append_jsonl(filename, record):
    _ensure_log_dir()
    path = os.path.join(LOG_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return path


def log_failure(stage, case_id, reason, payload=None):
    """
    Record something that needs manual review. Never raises on the caller's
    behalf for data reasons -- this is the "flag it, don't drop it silently" path.

    stage: short string identifying which check failed, e.g.
        "build_test_set", "held_out_patch", "held_out_preflight",
        "retrieval_leak", "judge_parse"
    case_id: the row/case/exemplar id this failure is about
    reason: human-readable explanation
    payload: raw offending data, for manual inspection
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "case_id": case_id,
        "reason": reason,
        "payload": payload,
    }
    path = _append_jsonl(f"{stage}_failures.jsonl", record)
    return path


def log_fix(stage, case_id, reason, payload=None):
    """
    Record something that was detected as broken AND automatically repaired
    (e.g. preflight autofix, retrieval retry recovery). Kept separate from
    log_failure so summaries can distinguish "needs your eyes" from
    "self-healed, here's a record of it."
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "case_id": case_id,
        "reason": reason,
        "payload": payload,
    }
    path = _append_jsonl(f"{stage}_fixes.jsonl", record)
    return path


def read_jsonl(filename):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_jsonl(filename):
    return len(read_jsonl(filename))


# ---------------------------------------------------------------------------
# Band-bin / Overall-score validation
#
# NOTE: per the real ingest.py, `band_bin` in Chroma metadata is a derived
# CATEGORICAL string (poor/developing/competent/expert), computed from the
# raw `Overall` float via compute_band_bin(). The "multiples of 0.5 between
# 1.0 and 9.0" rule validates `Overall`, not `band_bin` -- these are two
# different fields. Linsey is moving this constant into core/config.py as
# the shared source of truth; import it from there instead of redefining it
# here. See the snippet suggested alongside this file for what to add to
# core/scoring_utils.py (VALID_OVERALL_SCORES + validate_overall_score, and
# optionally compute_band_bin so build_test_set.py's stratification uses
# the exact same bucketing as ingest.py rather than a reimplementation that
# could drift out of sync).
# ---------------------------------------------------------------------------

import sys as _sys

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # D:\BandIt
sys.path.insert(0, str(ROOT))          # so "src.core.X" resolves
sys.path.insert(0, str(ROOT / "src"))  # so "core.X" resolves too

from src.core.scoring_utils import VALID_OVERALL_SCORES, validate_overall_score  # noqa: E402


# ---------------------------------------------------------------------------
# Chroma metadata helpers -- read-merge-write, since collection.update()
# REPLACES the entire metadata dict for an id rather than merging into it.
# Patching held_out=True naively (without reading current metadata first)
# would silently wipe out band_bin/overall on that record.
# ---------------------------------------------------------------------------

def chroma_get_metadata(collection, id_):
    """Returns the metadata dict for one id, or None if not found."""
    result = collection.get(ids=[id_])
    metadatas = result.get("metadatas") or []
    if not metadatas or metadatas[0] is None:
        return None
    return metadatas[0]


def chroma_update_metadata(collection, id_, **fields):
    """Merges `fields` into the existing metadata for id_, then writes it back."""
    current = chroma_get_metadata(collection, id_) or {}
    merged = {**current, **fields}
    collection.update(ids=[id_], metadatas=[merged])