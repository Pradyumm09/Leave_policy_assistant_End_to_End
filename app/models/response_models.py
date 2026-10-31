from pydantic import BaseModel


class Source(BaseModel):
    id: str | None = None
    section: str | None = None
    type: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float | None = None
    llm_provider: str | None = None
    intent: str | None = None
    retrieval_method: str | None = None