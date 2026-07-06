from pymilvus import MilvusClient

from config import setting
import logging

logger = logging.getLogger(__name__)

class MilvusVectorClient:
    def __init__(self,
        collection_name: str = None,
        colllection_dim: int = 4096,
        db_directory: str = None,
    ):
        self.collection_name = collection_name or setting.Milvus_collection_name
        self.db_directory = db_directory or setting.Milvus_db_directory
        from .embedding import embedding
        self.embedding_fn = embedding
        self.colllection_dim = colllection_dim

        logger.info(f"初始化向量数据库...")

        self.milvusvector = MilvusClient(
            str(self.db_directory),
        )

        if not self.milvusvector.has_collection(self.collection_name):
            self.milvusvector.create_collection(
                collection_name=self.collection_name,
                dimension=setting.EMBEDDING_DIM,
                metric_type="IP",
            )
        self.milvusvector.load_collection(self.collection_name)

_milvus_client = None


def get_milvus_client():
    global _milvus_client

    if _milvus_client is None:
        _milvus_client = MilvusVectorClient().milvusvector

    return _milvus_client


class LazyMilvusClient:
    def __getattr__(self, name):
        return getattr(get_milvus_client(), name)


milvusvector = LazyMilvusClient()
