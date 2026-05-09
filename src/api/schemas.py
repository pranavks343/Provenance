
"""Pydantic schemas for API requests, responses, and structured RAG output.

Field descriptions are read by the LLM via with_structured_output(),
so they double as instructions to the model — keep them precise.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single source reference supporting a claim in the answer."""

    source: str = Field(description="The PDF filename (basename only).")
    page: int = Field(description="The 1-indexed page number.")
    quote: str = Field(
        description=(
            "A short verbatim excerpt (under 200 characters) from the "
            "source that supports the claim."
        ),
        max_length=200,
    )


class Answer(BaseModel):
    """A grounded answer with structured citations and self-assessment."""

    text: str = Field(description="The answer prose, written for the user.")
    citations: list[Citation] = Field(
        description=(
            "Sources backing each non-trivial claim. Empty list if no "
            "context was used."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high = answer fully supported by context; "
            "medium = partially supported; "
            "low = context insufficient or ambiguous."
        )
    )
    used_context: bool = Field(
        description="True if retrieved context actually informed the answer."
    )


class UploadResponse(BaseModel):
    """Response from POST /upload."""

    document_id: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    """Request body for /query and /structured-query."""

    document_id: str
    question: str = Field(min_length=1, max_length=2000)
