from fastapi import APIRouter

from app.schemas.text_schema import (
    TextRequest,
    KeywordsResponse,
    SentimentResponse,
    SummaryResponse,
)
from app.services import text_service

router = APIRouter(prefix="/text", tags=["text"])


@router.post("/keywords", response_model=KeywordsResponse)
def keywords(request: TextRequest) -> KeywordsResponse:
    extracted = text_service.extract_keywords(request.text)
    return KeywordsResponse(keywords=extracted)


@router.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: TextRequest) -> SentimentResponse:
    result = text_service.analyze_sentiment(request.text)
    return SentimentResponse(**result)


@router.post("/summary", response_model=SummaryResponse)
def summary(request: TextRequest) -> SummaryResponse:
    result = text_service.summarize_text(request.text)
    return SummaryResponse(summary=result)
