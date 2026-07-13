from __future__ import annotations

from typing import TypedDict, Literal
from pydantic import BaseModel
from schema.analyzer_schema import QueryAnalysis
from schema.evaluator_schema import RetrievalEvaluation
from schema.researcher_schema import ResearchResult

from typing_extensions import NotRequired

class AgentState(TypedDict):
    # 输入
    query: str
    # analysis: QueryAnalysis
    analysis: QueryAnalysis

    # 检索
    # retrieved_nodes: list
    retrieved_nodes: NotRequired[list]
    # retrieval_evaluation: RetrievalEvaluation
    retrieval_evaluation: NotRequired[RetrievalEvaluation]

    # 缺失知识
    # missing_papers: list[str]
    # acquisition_queries: list[str]

    # 外部获取
    # resolved_papers: list[PaperResolution]
    # pdf_paths: list[str]
    # ingested_paper_ids: list[str]

    # 流程控制
    retrieval_round: int
    # arxiv_retry_count: int
    # errors: list[str]

    # 输出
    research_result: NotRequired[ResearchResult]
    answer: NotRequired[str]

class PaperResolution(BaseModel):
    paper_name: str
    source: Literal["zotero", "arxiv"] | None = None
    status: Literal[
        "pending",
        "resolved",
        "not_found",
        "processing_failed"
    ]
    pdf_path: str | None = None
    error: str | None = None
