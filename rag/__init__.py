"""RAG components, loaded lazily to avoid initializing models on package import."""

from __future__ import annotations

from typing import Any

__all__ = [
    "splitter",
    "embedding",
    "milvusvector",
]


def __getattr__(name: str) -> Any:
    if name == "embedding":
        from .embedding import embedding

        return embedding
    if name == "splitter":
        from .parser.splitter import splitter

        return splitter
    if name == "milvusvector":
        from .vector import MilvusHybridClient

        client = MilvusHybridClient()
        globals()[name] = client
        return client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
