from __future__ import annotations

from typing import Any

from schema.analyzer_schema import QueryAnalysis
from rag.embedding import Embedding
from rag.vector import MilvusVectorClient


class PaperRetriever:
    def __init__(self):
        self.vector_store = MilvusVectorClient()
        self.embed_model = Embedding()

    async def retrieve(
        self,
        query: str,
        analysis: QueryAnalysis,
        top_k: int = 5,
    ) -> list[Any]:
        query_text = self._build_query(query, analysis)
        filters = self._build_filters(analysis)

        results = await self.vector_store.search(
            query_text=query_text,
            embed_model=self.embed_model,
            top_k=top_k,
            filters=filters,
        )

        return results

    def _build_query(self, query: str, analysis: QueryAnalysis) -> str:
        parts = [
            query,
            *analysis.entities,
            *analysis.keywords,
        ]

        # 去重并过滤空字符串
        return " ".join(
            dict.fromkeys(
                part.strip()
                for part in parts
                if part and part.strip()
            )
        )

    def _build_filters(self, analysis: QueryAnalysis) -> dict[str, list[str]] | None:
        filters: dict[str, list[str]] = {}

        if analysis.paper_names:
            filters["paper_id"] = [name.lower() for name in analysis.paper_names]

        if analysis.section_types:
            filters["section_type"] = [
                item.value if hasattr(item, "value") else item
                for item in analysis.section_types
            ]

        return filters or None
