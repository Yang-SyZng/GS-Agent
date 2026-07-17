from pydantic import BaseModel, Field


class RetrievalEvaluation(BaseModel):
    missing_papers: list[str | None] = Field(
        description="The papers that are missing or failed to be retrieved in the knowledge base when available"
    )

    missing_information: list[str] = Field(
        description="The information needed to address the remaining issues that are still missing from the recall."
    )

    relevant_chunk_ids: list[str] = Field(
        description="Chunk ID relevant to the question and usable for answering"
    )

    sufficient: bool = Field(
        description="Is the recalled content sufficient to answer the user's questions?"
    )

    reason: str = Field(
        description="Short reasons for determining whether something is sufficient or insufficient。"
    )

