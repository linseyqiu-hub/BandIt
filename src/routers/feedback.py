"""
BandIt feedback router
POST /feedback — generates criterion-aligned IELTS examiner feedback
"""

import logging

from fastapi import APIRouter, Request, HTTPException

from schemas.requests import FeedbackRequest
from schemas.responses import FeedbackResponse
from services.feedback import generate_feedback


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: Request, body: FeedbackRequest):
    # Plain `def`, not `async def`: generate_feedback makes BLOCKING calls
    # (embedding, Chroma query, a multi-second Claude call). FastAPI runs a
    # `def` endpoint on a worker thread, so the event loop stays free for
    # other requests. Inside `async def` these calls would freeze the loop
    # and serialize every request. (Planned: full async stack — see to-do.)
    try:
        feedback_text = generate_feedback(
            question  = body.question,
            essay     = body.essay,
            scores    = body.scores,
            tone      = body.tone,
            app_state = request.app.state,
        )
    except Exception as e:
        # Every failure here is server-side (embedding, Chroma, Claude): input
        # was already validated by pydantic (422) before this route ran.
        # Log type + traceback (message redacted by the formatter) and a safe
        # status_code when the error has one (Anthropic API errors do).
        logger.exception(
            "feedback failed error=%s status_code=%s",
            type(e).__name__, getattr(e, "status_code", "-"),
        )
        # The client gets a stable code + x-request-id, never internals.
        raise HTTPException(status_code=500, detail="FEEDBACK_ERROR")

    return FeedbackResponse(feedback=feedback_text)