from pydantic import BaseModel


class Scores(BaseModel):
    """
    The 4 predicted sub-scores + computed Overall.
    All values are IELTS band scores: 1.0–9.0 in 0.5 increments.
    """
    task_response:               float
    coherence_cohesion:          float
    lexical_resource:            float
    grammatical_range_accuracy:  float
    overall:                     float


class Meta(BaseModel):
    """
    Metadata about the request — not scores, but useful for the frontend.
    """
    word_count: int
    truncated:  bool   # True if essay exceeded 512 tokens and was tail-truncated


class ScoreResponse(BaseModel):
    """
    Full response body for POST /score.
    """
    scores: Scores
    meta:   Meta


class ErrorDetail(BaseModel):
    """
    Envelope for all error responses.
    Consistent shape so the frontend always knows where to look.
    """
    code:    str   # machine-readable, e.g. "ESSAY_TOO_SHORT"
    message: str   # human-readable, e.g. "Essay must be at least 50 words. Received 4 words."


class ErrorResponse(BaseModel):
    error: ErrorDetail

class FeedbackResponse(BaseModel):
    feedback: str