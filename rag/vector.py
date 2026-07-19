from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.milvus import MilvusVectorStore

from config import setting

logger = logging.getLogger(__name__)


def _bypass_proxy_for_local_milvus(uri: str) -> None:
    """绕过代理"""

    hostname = urlparse(uri).hostname
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        return
    required = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        os.environ[key] = ",".join(dict.fromkeys([*existing, *required]))


class MilvusHybridClient:
    def __init__(
        self,
        collection_name: str | None = None,
        collection_dim: int | None = None,
        uri: str | None = None,
    ) -> None:
        self.collection_name = collection_name or setting.Milvus_collection_name
        self.collection_dim = collection_dim or setting.EMBEDDING_DIM
        self.uri = uri or setting.Milvus_uri
        _bypass_proxy_for_local_milvus(self.uri)

        logger.info(
            "Init Milvus hybrid retrieve: uri=%s, collection=%s, ranker=%s",
            self.uri,
            self.collection_name,
            setting.Milvus_hybrid_ranker,
        )
        self.vector_store = MilvusVectorStore(
            uri=self.uri,
            token=setting.Milvus_token,
            collection_name=self.collection_name,
            dim=self.collection_dim,
            enable_dense=True,
            enable_sparse=True,
            hybrid_ranker=setting.Milvus_hybrid_ranker,
            hybrid_ranker_params=self._ranker_params(),
            similarity_metric="IP",
            overwrite=False,
        )

    def _ranker_params(self) -> dict[str, Any]:
        if setting.Milvus_hybrid_ranker == "WeightedRanker":
            return {"weights": setting.Milvus_hybrid_weights}
        return {"k": setting.Milvus_rrf_k}

    def add_documents(self, nodes: list[dict[str, Any] | TextNode]) -> list[str]:
        """text -> BM25 sparse vectors"""

        text_nodes = [self._to_text_node(node) for node in nodes]
        if not text_nodes:
            return []
        ids = self.vector_store.add(text_nodes)
        logger.info("成功添加 %d 个文档", len(ids))
        return ids

    async def search(
        self,
        query_text: str,
        *,
        embed_model: Any,
        filters: dict[str, list[str]] | None = None,
        top_k: int = 5,
    ) -> list[Any]:
        """Native dense + BM25 retrieval and Milvus rank fusion."""

        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=embed_model,
        )
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=top_k,
            filters=self._metadata_filters(filters),
        )
        return await retriever.aretrieve(query_text)

    @staticmethod
    def _to_text_node(node: dict[str, Any] | TextNode) -> TextNode:
        if isinstance(node, TextNode):
            return node
        return TextNode(
            id_=str(node.get("id_") or node.get("id")),
            text=str(node.get("text") or ""),
            metadata=node.get("metadata") or {},
            embedding=node.get("embedding"),
        )

    @staticmethod
    def _metadata_filters(
        filters: dict[str, list[str]] | None,
    ) -> MetadataFilters | None:
        if not filters:
            return None
        grouped_filters = [
            MetadataFilter(key=key, value=values, operator=FilterOperator.IN)
            for key, values in filters.items()
        ]
        return MetadataFilters(
            filters=grouped_filters,
            condition=FilterCondition.AND,
        )


MilvusVectorClient = MilvusHybridClient
