from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

from enum import Enum

class PaperInfo(BaseModel):
    paper_id:str
    title:str
    authors:list[str]
    year:int
    pdf_path:str
    abstract:str

class SectionNode(BaseModel):
    section_id: str | None = None
    # 文档结构信息
    level: int | None = None
    path: str | None = None
    # 语义分类
    semantic_type: str | None = None
    title: str | None = None
    content: str | None = None
    children: List[SectionNode] = []

class SectionType(Enum):
    ABSTRACT="abstract"
    KEYWORDS="keywords"
    INTRODUCTION="introduction"
    BACKGROUND="background"
    RELATED_WORK="related_work"
    METHOD="method"
    EXPERIMENT="experiment"
    IMPLEMENTATION="implementation"
    EXPERIMENT_SETUP="experiment_setup"
    RESULTS="results"
    ABLATION="ablation"
    CONCLUSION="conclusion"
    REFERENCE="reference"
    APPENDIX="appendix"
    OTHER="other"

SECTION_KEYWORDS = {
    SectionType.ABSTRACT: [
        "abstract"
    ],
    SectionType.KEYWORDS: [
        "keyword",
        "keywords"
    ],
    SectionType.INTRODUCTION: [
        "introduction",
        "overview"
    ],
    SectionType.RELATED_WORK:[
        "related work",
        "related works",
        "background",
        "preliminary"
    ],
    SectionType.METHOD:[
        "method",
        "methodology",
        "approach",
        "framework",
        "preliminary"
    ],
    SectionType.EXPERIMENT:[
        "experiments",
        "experiment",
        "evaluation",
        "result",
        "benchmark"
    ],
    SectionType.CONCLUSION:[
        "conclusion",
        "discussion",
        "future work"
    ],
    SectionType.REFERENCE:[
        "reference",
        "references",
        "bibliography"
    ],
    SectionType.APPENDIX:[
        "appendix",
        "supplementary materials",
        "additional results"
    ],

}
CHILD_KEYWORDS = {
    SectionType.BACKGROUND:[
        "preliminary",
        "background",
        "overview",
        "notation",
        "problem formulation"
    ],
    SectionType.EXPERIMENT:[
        "dataset",
        "evaluation",
        "benchmark",
        "ablation",
        "comparison",
        "implementation details"
    ]
}

# {
# "id": str,

# "embedding": vector,

# "text": str,


# "paper_id": str,

# "section_id": str,

# "section_path": list[str],

# "section_type": str,

# "year": int
# }