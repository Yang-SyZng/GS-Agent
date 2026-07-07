import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence

from pymilvus import MilvusClient
from llama_index.core.schema import BaseNode
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
        from .embedding import embedding
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

    @property
    def client(self) -> MilvusClient:
        return self.milvusvector

    def __getattr__(self, name: str):
        return getattr(self.milvusvector, name)

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
        if node.embedding is None:
            raise ValueError(f"节点 {node.node_id} 缺少 embedding，请先调用 embedding.embed_nodes(nodes)")

        node_id = str(node.node_id)
        row_id: str | int = node_id if self._use_string_id else self._node_id_to_int(node_id)

        return {
            "id": row_id,
            "vector": node.embedding,
            "node_id": node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
            "node_json": json.dumps(node.to_dict(), ensure_ascii=False),
        }

    def add_documents(
        self,
        nodes: Sequence[BaseNode],
    ) -> list[Any]:
        if isinstance(nodes, BaseNode):
            nodes = [nodes]

        missing_embeddings = [node for node in nodes if node.embedding is None]
        if missing_embeddings:
            self.embedding_fn.embed_nodes(missing_embeddings)

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

_milvus_vector_client: MilvusVectorClient | None = None


def get_milvus_vector_client() -> MilvusVectorClient:
    global _milvus_vector_client

    if _milvus_vector_client is None:
        _milvus_vector_client = MilvusVectorClient()

    return _milvus_vector_client


def get_milvus_client() -> MilvusClient:
    return get_milvus_vector_client().client


class LazyMilvusClient:
    @property
    def collection_name(self) -> str:
        return get_milvus_vector_client().collection_name

    @property
    def db_directory(self) -> str | Path:
        return get_milvus_vector_client().db_directory

    @property
    def collection_dim(self) -> int:
        return get_milvus_vector_client().collection_dim

    @property
    def client(self) -> MilvusClient:
        return get_milvus_vector_client().client

    def add_documents(self, nodes: Sequence[BaseNode]) -> list[Any]:
        return get_milvus_vector_client().add_documents(nodes)

    def insert(self, *args: Any, **kwargs: Any):
        return get_milvus_client().insert(*args, **kwargs)

    def search(self, *args: Any, **kwargs: Any):
        return get_milvus_client().search(*args, **kwargs)

    def query(self, *args: Any, **kwargs: Any):
        return get_milvus_client().query(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any):
        return get_milvus_client().delete(*args, **kwargs)

    def upsert(self, *args: Any, **kwargs: Any):
        return get_milvus_client().upsert(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(get_milvus_vector_client(), name)


milvusvector = LazyMilvusClient()
