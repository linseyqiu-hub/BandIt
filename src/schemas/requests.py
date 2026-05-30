from pydantic import BaseModel


class ScoreRequest(BaseModel):
    """
    Body for POST /score.

    Both fields are required — FastAPI returns 422 automatically
    if either is missing or the wrong type.
    Deeper validation (word count, whitespace) happens in services/scoring.py.
    """
    question: str
    essay: str