import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence, Dict

from pymilvus import MilvusClient, DataType, Function, FunctionType
from llama_index.core.schema import BaseNode
from llama_index.core.tools import FunctionTool

from .embedding import embedding

from config import setting
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MilvusVectorClient:
    def __init__(self,
        collection_name: str = None,
        collection_dim: int = None,
        db_directory: str = None,
    ):
        self.collection_name = collection_name or setting.Milvus_collection_name
        self.db_directory = db_directory or setting.Milvus_db_directory
        self.embedding_fn = embedding
        self.collection_dim = collection_dim or setting.EMBEDDING_DIM

        logger.info(f"初始化向量数据库...")

        self.milvusvector = MilvusClient(
            str(self.db_directory),
        )

        if not self.milvusvector.has_collection(self.collection_name):
            self.milvusvector.create_collection(
                collection_name=self.collection_name,
                dimension=self.collection_dim,
                id_type="string",
                max_length=128,
                metric_type="IP",
            )
        self.milvusvector.load_collection(self.collection_name)
        self._use_string_id = self._collection_uses_string_id()

    def _collection_uses_string_id(self) -> bool:
        description = self.milvusvector.describe_collection(self.collection_name)
        for field in description.get("fields", []):
            if field.get("name") != "id":
                continue
            field_type = str(field.get("type", "")).upper()
            return "VARCHAR" in field_type or "STRING" in field_type
        return False

    def _node_id_to_int(self, node_id: str) -> int:
        digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

    def _node_to_row(self, node: BaseNode) -> dict[str, Any]:
        if node['embedding'] is None:
            raise ValueError(f"节点 {node.node_id} 缺少 embedding，请先调用 embedding.embed_nodes(nodes)")

        # node_id = str(node.node_id)
        # row_id: str | int = node_id if self._use_string_id else self._node_id_to_int(node_id)

        return {
            "id": node['id_'],
            "vector": node['embedding'],
            # "node_id": node_id,
            "text": node['text'],
            "metadata": node['metadata'],
            # "node_json": json.dumps(node.to_dict(), ensure_ascii=False),
        }

    def add_documents(
        self,
        nodes: List[Dict],
    ) -> list[Any]:
        # if isinstance(nodes, List[Dict]):
        #     nodes = [nodes]

        # missing_embeddings = [node for node in nodes if node.embedding is None]
        # if missing_embeddings:
        #     self.embedding_fn.embed_nodes(missing_embeddings)

        rows = [self._node_to_row(node) for node in tqdm(nodes, desc="processing")]
        if not rows:
            return []

        result = self.milvusvector.insert(
            collection_name=self.collection_name,
            data=rows,
        )
        ids = result.get("ids", [])
        logger.info(f"成功添加{len(ids)}个文档")
        return ids

    def search(
        self,
        query: Sequence[float],
        filters: str | dict[str, list[str]] | None = None,
        top_k: int = 5,
    ):
        if isinstance(query, (str, bytes)):
            raise TypeError("query must be an embedding vector, not text")

        milvus_filter = filters if isinstance(filters, str) else None
        limit = top_k if milvus_filter else max(top_k * 10, top_k)
        res = self.milvusvector.search(
            collection_name=self.collection_name,
            data=[list(query)],
            limit=limit,
            output_fields=["text", "metadata"],
            filter=milvus_filter,
        )

        hits = res[0]
        if isinstance(filters, dict):
            hits = [hit for hit in hits if self._matches_metadata_filters(hit, filters)]

        return hits[:top_k]

    def _matches_metadata_filters(
        self,
        hit: Any,
        filters: dict[str, list[str]],
    ) -> bool:
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        metadata = entity.get("metadata", {})

        for key, allowed_values in filters.items():
            value = metadata.get(key)
            if value not in allowed_values:
                return False

        return True


milvusvector = MilvusVectorClient()

ragtools = [
    FunctionTool.from_defaults(
        fn=milvusvector.search
        ),
]
