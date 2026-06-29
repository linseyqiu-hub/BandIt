from pydantic import BaseModel
from schemas.responses import Scores

class ScoreRequest(BaseModel):
    """
    Body for POST /score.

    Both fields are required — FastAPI returns 422 automatically
    if either is missing or the wrong type.
    Deeper validation (word count, whitespace) happens in services/scoring.py.
    """
    question: str
    essay: str


class FeedbackRequest(BaseModel):
    question: str
    essay:    str
    scores:   Scores
    tone:     str = "coaching"   # default coaching, override with "strict"