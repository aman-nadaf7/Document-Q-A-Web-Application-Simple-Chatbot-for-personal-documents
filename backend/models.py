from pydantic import BaseModel, field_validator
from typing import List


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int


class AskRequest(BaseModel):
    document_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question must not be empty")
        return v.strip()


class AskResponse(BaseModel):
    answer: str
    sources: List[int]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str