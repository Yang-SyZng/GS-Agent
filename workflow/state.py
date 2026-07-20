from __future__ import annotations

from typing import Any, TypedDict, Literal
from pydantic import BaseModel
from schema.analyzer_schema import QueryAnalysis
from schema.evaluator_schema import RetrievalEvaluation

from typing_extensions import NotRequired

class AgentState(TypedDict):
    # 输入
    user_query: str
    # analysis: QueryAnalysis
    analysis: QueryAnalysis

    # 检索
    retrieved_nodes: NotRequired[list]
    retrieval_evaluated_result: NotRequired[RetrievalEvaluation]

    # 外部搜索（本地知识库中不存在目标材料时）
    external_search_results: NotRequired[list[Any]]
    external_search_errors: NotRequired[list[str]]
    paper_resolutions: NotRequired[list[PaperResolution]]
    ingested_paper_ids: NotRequired[list[str]]
    knowledge_updated: NotRequired[bool]


    # 流程控制
    ## 查询索引次数
    retrieval_round: int = 0
    external_search_round: int = 0


    # 输出
    answer: NotRequired[str]

class PaperResolution(BaseModel):
    paper_name: str
    paper_id: str | None = None
    title: str | None = None
    source: Literal["zotero", "arxiv"] | None = None
    status: Literal[
        "pending",
        "resolved",
        "not_found",
        "processing_failed"
    ]
    pdf_path: str | None = None
    chunks_indexed: int = 0
    error: str | None = None
