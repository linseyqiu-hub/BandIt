"""
core/constants.py — values that are the same in every environment.

Rules for this module:
    - No environment access, no I/O, no imports from the rest of the app.
    - Importing it can never fail, so any module may import it freely.

If a value differs between laptop / compose / AWS, it does NOT belong here —
it belongs in core/settings.py.
"""

# ── Scoring model identity ──────────────────────────────────────────────
# The base architecture and tokenizer the checkpoint was trained on.
# This is part of the model's definition, not deployment config: changing it
# without retraining breaks the checkpoint.
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
LABEL_COLUMNS = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]
VALID_OVERALL_SCORES = frozenset(round(1.0 + 0.5 * i, 1) for i in range(17))

# ── Retrieval embedding model ───────────────────────────────────────────
# Single source of truth for the embedding model. Ingest WRITES vectors with
# it and serving/eval QUERY with it — if these ever differ, retrieval silently
# degrades with no error. Previously hardcoded separately in lifespan.py,
# scripts/ingest.py and eval/run_ragas.py.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Request validation rules ────────────────────────────────────────────
ESSAY_MIN_WORDS = 50
ESSAY_MAX_WORDS = 1200
QUESTION_MIN_CHARS = 10