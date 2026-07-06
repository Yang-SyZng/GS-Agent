from __future__ import annotations

__all__ = [
    "splitter",
    "embedding",
    "get_milvus_client",
    "milvusvector",
]

def __getattr__(name: str):
    if name in {"splitter"}:
        from .splitter import splitter
        
        return {
            "splitter": splitter,
        }[name]
    
    if name in {"embedding"}:
        from .embedding import embedding

        return {
            "embedding": embedding,
        }[name]

    if name in {"get_milvus_client", "milvusvector"}:
        from .vector import get_milvus_client, milvusvector

        return {
            "get_milvus_client": get_milvus_client,
            "milvusvector": milvusvector,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
