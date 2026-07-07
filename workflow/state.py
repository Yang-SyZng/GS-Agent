from typing import TypedDict, List, Optional, Dict, Any


class ResearchState(TypedDict):
    # Search
    query: str
    max_result: int
    papers: List[Dict[str, Any]]

    # Download
    selected_paper_id: Optional[str]
    pdf_path: Optional[str]

    # Parse
    extracted_text: Optional[str]
    text_length: Optional[int]

    # Extract

    # Index
    documents: List[str]  # 文档对象列表
    vector_ids: List[str]  # 向量ID列表
    indexed: bool  # 是否已索引

    # QA
    question: Optional[str]  # 用户问题
    retrieved_docs: List[str]  # 检索到的文档
    context: Optional[str]  # 上下文
    answer: Optional[str]  # 最终答案

