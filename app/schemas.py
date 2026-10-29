from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=500_000)

    @field_validator("document_id", "title")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    chunk_id: int
    document_id: str
    title: str
    page_number: int | None
    content: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
