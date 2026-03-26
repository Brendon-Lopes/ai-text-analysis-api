from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class KeywordsResponse(BaseModel):
    keywords: list[str]


class SentimentResponse(BaseModel):
    sentiment: str
    score: float


class SummaryResponse(BaseModel):
    summary: str
