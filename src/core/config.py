import os

# ── Model identity (unchanged) ──
MODEL_NAME     = "microsoft/deberta-v3-base"
MAX_LENGTH     = 512
LABEL_COLUMNS  = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]
VALID_OVERALL_SCORES = frozenset(round(1.0 + 0.5 * i, 1) for i in range(17))

# ── Model registry (HuggingFace Hub) ──
MODEL_REPO    = os.environ.get("MODEL_REPO", "burgerFlipperF/bandit-scorer")
MODEL_VERSION = os.environ["MODEL_VERSION"]   # required — wrong default is worse than crashing
HF_HOME       = os.environ.get("HF_HOME", "/models")

# ── ChromaDB ──
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

# ── Redis ──
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ── API behaviour ──
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# ── Validation rules (unchanged) ──
ESSAY_MIN_WORDS    = 50
ESSAY_MAX_WORDS    = 1200
QUESTION_MIN_CHARS = 10

# ── Secrets — validate at import time ──
REQUIRED_ENV = ["ANTHROPIC_API_KEY", "HF_TOKEN"]

def validate_env():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required env vars: {missing}")

validate_env()