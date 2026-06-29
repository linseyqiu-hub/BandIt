from fastapi import APIRouter, Request

from schemas.requests import ScoreRequest
from schemas.responses import ScoreResponse
from services.scoring import score_essay


router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
async def score(request: Request, body: ScoreRequest) -> ScoreResponse:
    """
    POST /score

    Scores an IELTS Task 2 essay across 4 criteria.
    Overall is computed server-side as the mean of the 4 sub-scores,
    rounded to the nearest 0.5.

    This handler does three things only:
        1. Extract the engine from app.state
        2. Hand off to the service
        3. Return the result

    All validation and business logic lives in services/scoring.py.
    """
    engine = request.app.state.inference_engine

    return score_essay(
        question = body.question,
        essay    = body.essay,
        engine   = engine,
    )