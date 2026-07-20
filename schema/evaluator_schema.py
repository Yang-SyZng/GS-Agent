from enum import Enum

from pydantic import BaseModel, Field


class RetrievalStatus(str, Enum):
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient"
    SUFFICIENT = "sufficient"


class RetrievalEvaluation(BaseModel):
    status: RetrievalStatus = Field(
        description=(
            "Retrieval route: not_found when the requested source is absent, "
            "insufficient when relevant sources exist but coverage is incomplete, "
            "or sufficient when the evidence can answer the query."
        )
    )

    missing_papers: list[str | None] = Field(
        description="The papers that are missing or failed to be retrieved in the knowledge base when available"
    )

    missing_information: list[str] = Field(
        description="The information needed to address the remaining issues that are still missing from the recall."
    )

    relevant_chunk_ids: list[str] = Field(
        description="Chunk ID relevant to the question and usable for answering"
    )

    reason: str = Field(
        description="Short reasons for determining whether something is sufficient or insufficient。"
    )

    @property
    def sufficient(self) -> bool:
        """Compatibility helper for callers that still need a boolean check."""
        return self.status == RetrievalStatus.SUFFICIENT
