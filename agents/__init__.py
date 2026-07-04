from __future__ import annotations

from .ZoteroAgent import ZoteroAgent

__all__ = [
    "MainAgent",
    "DocumentRouterAgent",
    "ZoteroAgent",
]


def __getattr__(name: str):
    if name == "MainAgent":
        from .MainAgent import MainAgent

        return MainAgent

    if name == "DocumentRouterAgent":
        from .DocumentRouterAgent import DocumentRouterAgent

        return DocumentRouterAgent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
