from __future__ import annotations
from typing import List

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core.schema import BaseNode

from config import setting
import logging

logger = logging.getLogger(__name__)


class Embedding(OpenAILikeEmbedding):
    def __init__(self,
        model_name: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        model_name = model_name or setting.EMBEDDING_MODEL_ID
        logger.info(f"初始化Embedding模型: {model_name}")

        super().__init__(
            model_name=model_name,
            api_key=api_key or setting.API_KEY,
            api_base=api_base or setting.BASE_URL,
        )

    def embed_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        texts = [node.get_content() for node in nodes]
        embeddings = self.get_text_embedding_batch(texts, show_progress=True)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    def embed_text(self, text: list[str] | str):
        try:
            if isinstance(text, str):
                text = [text]
            logger.info(f"正在生成text的embedding: {text[:50]}")
            return self.get_text_embedding_batch(text)
        except Exception as e:
            logger.error(f"text embedding错误: {str(e)}")
            raise
        
    def embed_query(self, text: str) -> List[float]:
        try:
            logger.debug(f"正在生成查询embedding: {text[:50]}...")
            return self.get_query_embedding(text)
        except Exception as e:
            logger.error(f"查询embedding错误: {str(e)}")
            raise
    
_embedding: Embedding | None = None


def get_embedding() -> Embedding:
    global _embedding

    if _embedding is None:
        _embedding = Embedding()

    return _embedding


class LazyEmbedding:
    def embed_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        return get_embedding().embed_nodes(nodes)

    def embed_text(self, text: list[str] | str):
        return get_embedding().embed_text(text)

    def embed_query(self, text: str) -> List[float]:
        return get_embedding().embed_query(text)

    def __getattr__(self, name: str):
        return getattr(get_embedding(), name)


embedding = LazyEmbedding()
