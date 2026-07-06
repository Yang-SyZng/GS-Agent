from __future__ import annotations
import os

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from config import setting
import logging
from typing import List

logger = logging.getLogger(__name__)


class Embedding:
    def __init__(self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        check_embedding_ctx_length: bool = False, # default False
    ):
        self.model = model or setting.EMBEDDING_MODEL_ID
        self.api_key = api_key or setting.API_KEY
        self.base_url = base_url or setting.BASE_URL

        logger.info(f"初始化Embedding模型: {self.model}")

        self.embedding = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            check_embedding_ctx_length=check_embedding_ctx_length,
        )

    
    @retry(
    stop=stop_after_attempt(setting.max_retries),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
    )
    def embed_text(self, text: list[str] | str):
        try:
            if isinstance(text, str):
                text = [text]
            logger.info(f"正在生成text的embedding: {text[:50]}")
            return self.embedding.embed_documents(text)
        except Exception as e:
            logger.error(f"text embedding错误: {str(e)}")
            raise
    

    @retry(
    stop=stop_after_attempt(setting.max_retries),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
    )
    def embed_query(self, text: str) -> List[float]:
        try:
            logger.debug(f"正在生成查询embedding: {text[:50]}...")
            return self.embedding.embed_query(text)
        except Exception as e:
            logger.error(f"查询embedding错误: {str(e)}")
            raise
    
_embedding = None


def get_embedding() -> Embedding:
    global _embedding

    if _embedding is None:
        _embedding = Embedding()

    return _embedding


class LazyEmbedding:
    def __getattr__(self, name):
        return getattr(get_embedding(), name)


embedding = LazyEmbedding()
