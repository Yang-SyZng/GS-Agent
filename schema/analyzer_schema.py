from pydantic import BaseModel, Field
from enum import Enum

class QueryType(str, Enum):
    # 针对一篇论文
    SINGLE_PAPER = "single_paper"
    # 针对多篇论文
    MULTI_PAPER = "multi_paper"
    # 一般搜索
    GENERAL_SEARCH = "general_search"

class QueryTarget(str, Enum):
    METHOD = "method"
    EXPERIMENT = "experiment"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    BACKGROUND = "background"
    OTHER = "other"

class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    CONCLUSION = "conclusion"
    REFERENCE = "reference"
    SUPPLEMENTARY = "supplementary"

class QueryAnalysis(BaseModel):
    query_type: QueryType = Field(
        description="The type of task for the user's question."
    )

    targets: list[QueryTarget] = Field(
        min_length=1,
        description="One or more of the information dimensions that the user is concerned about."
    )

    paper_names: list[str] = Field(
        description="The exact title of the paper that the user explicitly mentioned."
    )

    entities: list[str] = Field(
        min_length=3,
        description="Key entities such as methods, models, datasets, metrics, and technologies."
    )

    keywords: list[str] = Field(
        min_length=3,
        max_length=8,
        description="English keywords for retrieval"
    )

    section_types: list[SectionType] = Field(
        min_length=1,
        description="The recommended types of semantic chapters to be retrieved first"
    )
