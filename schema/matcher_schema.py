from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"


class PaperMatchResult(BaseModel):
    target_paper: str = Field(
        description="The paper title or identifier requested by the user"
    )

    paper_id: str | None = Field(
        default=None,
        description="Source-specific identifier of the matched candidate",
    )

    title: str | None = Field(
        default=None,
        description="Canonical title of the matched candidate",
    )

    authors: List[str | None] = Field(
        default=[],
        description="Authors of the matched candidate",
    )

    abstract: str | None = Field(
        default=None,
        description="Abstraction of the matched candidate",
    )

    doi: str | None = Field(
        default=None,
        description="DOI of the matched candidate when available"
    )

    published_date: str | None = Field(
        default=None,
        description="DOI of the matched paper when available",
    )

    pdf_url: str | None = Field(
        default=None,
        description="PDF document url of the matched candidate when available"
    )

    source: str | None = Field(
        default=None,
        description="source of the matched candidate when available"
    )

    categories: str | None = Field(
        default=None,
        description="categories of the matched candidate when available"
    )

    status: MatchStatus = Field(
        description="Whether one candidate reliably matches the requested paper"
    )

    candidate_index: int | None = Field(
        default=None,
        description="Zero-based index of the matched candidate, or null",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the metadata match",
    )

    reason: str = Field(
        description="A concise explanation based only on candidate metadata"
    )
