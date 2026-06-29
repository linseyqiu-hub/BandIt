import os


# ------------------------------------------------------------------
# Model identity
# ------------------------------------------------------------------

# HuggingFace model ID — must match what was used during training.
# Used by tokenizer (inference.py) and architecture init (model.py).
MODEL_NAME = "microsoft/deberta-v3-base"

# DeBERTa-v3-base processes at most 512 tokens per input.
# Essays exceeding this are truncated at the tail.
MAX_LENGTH = 512

# The 4 IELTS scoring criteria the model predicts.
# Order must match the column order in ielts_labeled.csv and
# the output neurons in the scoring head (Linear 768 → 4).
# Overall is NOT in this list — it is computed post-inference.
LABEL_COLUMNS = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# Absolute path to best_model.pt.
# os.path.dirname(__file__) is the core/ directory.
# We go up two levels to reach the project root, then into checkpoints/.
# This works regardless of where you run the server from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model_v5.pt")
MODEL_PATH = CHECKPOINT_PATH  # alias — lifespan.py imports this name
CHROMA_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma")


# ------------------------------------------------------------------
# Validation rules — single source of truth
# ------------------------------------------------------------------

# These are checked in services/scoring.py before any model call.
# Changing them here propagates everywhere automatically.

ESSAY_MIN_WORDS    = 50
ESSAY_MAX_WORDS    = 1200
QUESTION_MIN_CHARS = 10