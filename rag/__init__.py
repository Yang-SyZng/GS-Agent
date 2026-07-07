from __future__ import annotations

__all__ = [
    "splitter",
    "embedding",
    "get_milvus_vector_client",
    "get_milvus_client",
    "milvusvector",
]

def __getattr__(name: str):
    if name in {"splitter"}:
        from .splitter import splitter

        values = {
            "splitter": splitter,
        }
        globals().update(values)
        return values[name]
    
    if name in {"embedding"}:
        from .embedding import embedding

        values = {
            "embedding": embedding,
        }
        globals().update(values)
        return values[name]

    if name in {"get_milvus_vector_client", "get_milvus_client", "milvusvector"}:
        from .vector import get_milvus_client, get_milvus_vector_client, milvusvector

        values = {
            "get_milvus_vector_client": get_milvus_vector_client,
            "get_milvus_client": get_milvus_client,
            "milvusvector": milvusvector,
        }
        globals().update(values)
        return values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
