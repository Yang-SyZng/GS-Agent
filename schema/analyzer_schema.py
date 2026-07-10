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

class QueryAnalysis(BaseModel):

    original_query: str
    
    query_type: QueryType = Field(
        description="用户问题的类型"
    )

    target: QueryTarget = Field(
        description="用户关注的信息类型"
    )

    paper_names: list[str] = Field(
        default_factory=list,
        description="涉及的论文名称"
    )

    entities: list[str] = Field(
        default_factory=list,
        description="方法/模型/数据集等实体"
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="检索关键词"
    )

    section_types: list[str] = Field(
        default_factory=list,
        description="建议检索章节类型"
    )
