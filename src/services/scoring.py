from fastapi import HTTPException

from core.config import (
    ESSAY_MIN_WORDS,
    ESSAY_MAX_WORDS,
    QUESTION_MIN_CHARS,
)
from inference import BandItInferenceEngine
from schemas.responses import ScoreResponse, Scores, Meta


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_input(question: str, essay: str) -> None:
    """
    Runs all validation checks before the model is called.
    Raises HTTPException immediately on the first failure — short-circuits.

    Layers (in order):
        1. Whitespace check  — catches empty / blank inputs
        2. Question length   — must be meaningful enough for Task Response scoring
        3. Essay word count  — IELTS rules: minimum 50 words, we cap at 1200

    Pydantic already handled:
        - Missing fields (422 Unprocessable Entity)
        - Wrong types   (422 Unprocessable Entity)
    So we only check semantic rules here.

    Raises:
        HTTPException 400 with a code + message body on any failure.
    """

    # --- Layer 1: whitespace ---
    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_INPUT", "message": "Question cannot be empty or whitespace-only."}
        )
    if not essay.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_INPUT", "message": "Essay cannot be empty or whitespace-only."}
        )

    # --- Layer 2: question length ---
    if len(question.strip()) <= QUESTION_MIN_CHARS:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "QUESTION_TOO_SHORT",
                "message": f"Question must be more than {QUESTION_MIN_CHARS} characters."
            }
        )

    # --- Layer 3: essay word count ---
    word_count = len(essay.split())

    if word_count < ESSAY_MIN_WORDS:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "ESSAY_TOO_SHORT",
                "message": f"Essay must be at least {ESSAY_MIN_WORDS} words. Received {word_count} words."
            }
        )

    if word_count > ESSAY_MAX_WORDS:
        raise HTTPException(
            status_code=400,
            detail={
                "code":    "ESSAY_TOO_LONG",
                "message": f"Essay must be at most {ESSAY_MAX_WORDS} words. Received {word_count} words."
            }
        )


# ------------------------------------------------------------------
# Score
# ------------------------------------------------------------------

def score_essay(question: str, essay: str, engine: BandItInferenceEngine) -> ScoreResponse:
    """
    Validates input, runs inference, and returns a structured response.

    This is the only function routers/scoring.py calls.
    It knows nothing about HTTP — it takes plain strings, returns a Pydantic model.

    Args:
        question: IELTS task question string
        essay:    candidate essay string
        engine:   loaded BandItInferenceEngine from app.state

    Returns:
        ScoreResponse with scores + meta

    Raises:
        HTTPException 400 on validation failure
        HTTPException 500 on inference failure
    """

    # Run validation gates — raises 400 on first failure
    validate_input(question, essay)

    # Call the model — wrap in try/except so inference crashes
    # return a clean 500 instead of an ugly unhandled exception
    try:
        result = engine.score_with_metadata(question, essay)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code":    "INFERENCE_ERROR",
                "message": f"Model inference failed: {str(e)}"
            }
        )

    # Map engine output → response schema
    # engine returns keys like "Task_Response" (training column names)
    # API contract uses snake_case: "task_response"
    raw_scores = result["scores"]

    return ScoreResponse(
        scores=Scores(
            task_response              = raw_scores["Task_Response"],
            coherence_cohesion         = raw_scores["Coherence_Cohesion"],
            lexical_resource           = raw_scores["Lexical_Resource"],
            grammatical_range_accuracy = raw_scores["Range_Accuracy"],
            overall                    = raw_scores["Overall"],
        ),
        meta=Meta(
            word_count = result["word_count"],
            truncated  = result["truncated"],
        )
    )