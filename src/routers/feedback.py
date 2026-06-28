"""
BandIt feedback router
POST /feedback — generates criterion-aligned IELTS examiner feedback
"""

from fastapi import APIRouter, Request, HTTPException

from schemas.requests import FeedbackRequest
from schemas.responses import FeedbackResponse
from services.feedback import generate_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: Request, body: FeedbackRequest):
    try:
        feedback_text = generate_feedback(
            question  = body.question,
            essay     = body.essay,
            scores    = body.scores,
            tone      = body.tone,
            app_state = request.app.state,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="FEEDBACK_ERROR")

    return FeedbackResponse(feedback=feedback_text)